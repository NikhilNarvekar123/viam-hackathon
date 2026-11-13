"""Overview -> provisional XY -> optional individually confirmed hover. No gripper commands."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
import yaml
from robot.locate import locate_objects, annotate
from robot.viam_hw import connect, disconnect, load_credentials
from tools.inspect_wrist import capture
from robot.hover_motion import planned_hover


def convex_hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def map_point(uv, mapping):
    hull = convex_hull([m['overview_uv'] for m in mapping['markers']])
    if len(hull) < 3:
        raise ValueError('Calibration coverage is degenerate')
    u, v = uv
    for a, b in zip(hull, hull[1:]+hull[:1]):
        if (b[0]-a[0])*(v-a[1])-(b[1]-a[1])*(u-a[0]) < -1e-6:
            raise ValueError('Outside calibration coverage; refusing extrapolation')
    H = np.asarray(mapping['homography_overview_to_world_xy'], dtype=float)
    if H.shape != (3,3) or not np.isfinite(H).all():
        raise ValueError('Invalid mapping matrix')
    q = H @ [u,v,1]
    if abs(q[2]) < 1e-9:
        raise ValueError('Invalid projective denominator')
    xy = q[:2]/q[2]
    if not np.isfinite(xy).all():
        raise ValueError('Invalid mapped point')
    return xy.tolist()


def motion_targets(objects, settings):
    if settings['safety_verified'] is not True or settings['driver_speed_and_paths_verified'] is not True:
        raise ValueError('Real motion blocked: verify hover setup, driver speed, and swept paths first')
    z = settings['hover_endpoint_z_mm']
    offset = settings['inspection_point_minus_endpoint_xy_mm']
    bounds = settings['endpoint_xy_bounds_mm']
    orientation = settings['orientation']
    if z is None or offset is None or bounds is None or orientation is None:
        raise ValueError('Real motion blocked: missing measured height/orientation/offset/bounds')
    if len(offset)!=2 or len(bounds)!=4 or set(orientation)!={'o_x','o_y','o_z','theta'}:
        raise ValueError('Malformed motion configuration')
    values = [z,*offset,*bounds,*orientation.values(),settings['max_segment_mm']]
    if not np.isfinite(np.array(values,dtype=float)).all() or settings['max_segment_mm']<=0:
        raise ValueError('Motion values must be finite; segment limit must be positive')
    xmin,ymin,xmax,ymax = bounds
    if not xmin<xmax or not ymin<ymax:
        raise ValueError('Invalid endpoint workspace bounds')
    targets=[]
    for obj in objects:
        x,y = np.array(obj['world_xy_mm'])-offset
        if not xmin<=x<=xmax or not ymin<=y<=ymax:
            raise ValueError(f"Target {obj['id']} outside endpoint workspace")
        targets.append(dict(id=obj['id'],x=float(x),y=float(y),z=float(z)))
    return targets


async def execute(targets, cfg):
    from viam.components.arm import Arm
    from viam.proto.common import Pose
    settings=cfg['motion']
    machine=await connect(load_credentials(ROOT/'.env'))
    arm=None
    try:
        arm=Arm.from_robot(machine,cfg['arm_name'])
        current=await arm.get_end_position()
        ori=settings['orientation']
        for key in ['o_x','o_y','o_z','theta']:
            tolerance=1.0 if key=='theta' else 0.01
            if abs(getattr(current,key)-ori[key])>tolerance:
                raise ValueError('Arm must already have the verified inspection orientation; no automatic rotation')
        # First rise to the configured clearance at current XY; then hover-only XY travel.
        if current.z > targets[0]['z']+1:
            raise ValueError('Configured hover height is below current endpoint: refusing automatic descent')
        xmin,ymin,xmax,ymax=settings['endpoint_xy_bounds_mm']
        if not xmin<=current.x<=xmax or not ymin<=current.y<=ymax:
            raise ValueError('Current endpoint is outside configured safe workspace')
        waypoints=[dict(id='clearance',x=current.x,y=current.y,z=targets[0]['z'])]+targets
        previous=np.array([current.x,current.y,current.z])
        for w in waypoints:
            point=np.array([w[k] for k in ['x','y','z']])
            if np.linalg.norm(point-previous)>settings['max_segment_mm']:
                raise ValueError('A segment exceeds max_segment_mm; review setup rather than bypassing blindly')
            previous=point
        for w in waypoints:
            text=f"Move {w['id']} to endpoint XYZ ({w['x']:.1f}, {w['y']:.1f}, {w['z']:.1f}) mm? Type MOVE: "
            if await asyncio.to_thread(input,text)!='MOVE':
                print('Cancelled. Arm stays at its current position.');return
            await arm.move_to_position(Pose(x=w['x'],y=w['y'],z=w['z'],**ori))
            print('Reached',w['id'])
    except BaseException:
        if arm is not None:
            try:
                await asyncio.wait_for(arm.stop(),timeout=5)
            except Exception:
                print('Stop failed. Use physical emergency stop if needed.',file=sys.stderr)
        raise
    finally:
        await disconnect(machine)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/overview_hover.yaml')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--image',type=Path)
    source.add_argument('--fake',action='store_true',help='Synthetic camera; no real motion')
    parser.add_argument('--ids',nargs='+',type=int,help='Frame-local candidate IDs to visit')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--plan',action='store_true',help='Read live geometry and collision-check route WITHOUT moving')
    parser.add_argument('--confirm-motion',action='store_true')
    parser.add_argument('--accept-provisional-calibration',action='store_true')
    args=parser.parse_args()
    if args.execute and (args.image or args.fake or not args.ids or not args.confirm_motion or not args.accept_provisional_calibration):
        parser.error('Execution requires live capture, --ids, --confirm-motion, and --accept-provisional-calibration')
    try:
        cfg=yaml.safe_load(args.config.read_text())
        mapping=json.loads((ROOT/cfg['mapping']).read_text())
        loc=yaml.safe_load((ROOT/cfg['locator_config']).read_text())
        if args.image:
            with Image.open(args.image) as im: image=im.convert('RGB')
        else:
            image=asyncio.run(capture(cfg,args.fake))
        with Image.open(ROOT/cfg['background']) as im: background=im.convert('RGB')
        if image.size != (640,480):
            raise ValueError('This provisional mapping was captured at 640x480')
        result,mask=locate_objects(image,background,loc)
        usable=[]
        for obj in result['objects']:
            try:
                if obj['warnings']: raise ValueError('Boundary candidate')
                obj['world_xy_mm']=map_point(obj['center_uv'],mapping)
                usable.append(obj)
            except ValueError as exc:
                obj['mapping_rejection']=str(exc)
        folder=ROOT/'logs/overview_hover'/datetime.now(timezone.utc).strftime('hover_%Y%m%dT%H%M%S_%fZ')
        folder.mkdir(parents=True)
        image.save(folder/'frame.png')
        output=annotate(image,result,mask);draw=ImageDraw.Draw(output)
        for obj in usable:
            x,y=obj['world_xy_mm'];u,v=obj['center_uv']
            draw.text((max(0,u-70),v+10),f"#{obj['id']} XY {x:.0f}, {y:.0f} mm",fill='white',stroke_width=1,stroke_fill='black')
            print(f"Candidate {obj['id']}: pixels {obj['center_uv']}, provisional world XY {[round(x,1),round(y,1)]}")
        output.save(folder/'targets.png')
        result.update(mapping=cfg['mapping'],validated_for_motion=False)
        (folder/'targets.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print('Review:',folder/'targets.png')
        selected=usable if not args.ids else [o for o in usable if o['id'] in args.ids]
        if args.ids and {o['id'] for o in selected}!=set(args.ids):
            raise ValueError('Selected ID is missing or rejected')
        if not selected: print('No usable targets.');return
        if not args.execute and not args.plan:
            for obj in selected: print('SIMULATED visit',obj['id'],obj['world_xy_mm'])
            print('PREVIEW ONLY. No arm connected. Inspect for markers, clutter, merged objects and camera movement.')
            return
        if not args.ids:
            raise ValueError('Select specific --ids for planning or execution')
        asyncio.run(planned_hover(selected,cfg,folder,ROOT,execute=args.execute))
    except KeyboardInterrupt:
        parser.exit(130,'Cancelled.\n')
    except (ValueError,OSError,ImportError,KeyError,RuntimeError) as exc:
        parser.exit(1,f'Hover demo error: {exc}\n')


if __name__=='__main__':
    main()
