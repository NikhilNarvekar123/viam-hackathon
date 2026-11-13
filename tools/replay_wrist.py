"""Replay a SAVED frame through current wrist stages; never captures or moves hardware."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from PIL import Image
import yaml
from anomaly.wrist import WristInspector
from vlm.service import decide_wrist_objects


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--vlm',action='store_true',help='Run local Viam VLM inference on raw saved pixels (no motion)')
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'config/wrist_inspection.yaml').read_text())
    inspector=WristInspector(cfg,ROOT)
    with Image.open(args.image) as image: report=inspector.process(image)
    report['replay_source']=str(args.image.resolve())
    folder=Path(report['output_dir'])
    (folder/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.vlm:
        settings=yaml.safe_load((ROOT/'config/vlm_service.yaml').read_text())
        asyncio.run(decide_wrist_objects(report,settings,ROOT))
    print('REPLAY OUTPUT',folder)


if __name__=='__main__': main()
