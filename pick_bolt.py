#!/usr/bin/env python3
"""
pick_bolt.py - find a bolt on the table with the wrist camera and pick it up.

Station: armfarm22 (xArm6 `arm`, `gripper`, wrist RealSense `cam`, motion service `builtin`).

REPOSITORY INTEGRATION: standalone execution is READ-ONLY. Use
    python main.py --stage pick --confirm-motion
for classified picking. The upstream instructions/modes below are retained as
reference, not enabled standalone motion entrypoints in this checkout.

UPSTREAM: RUN IN THE VIAM APP (machine page -> CODE tab -> Python):
    Paste this whole file over main.py, set MODE below, press Run. Output appears in CONSOLE.
    The sandbox supplies the machine address and API key itself. There are no prompts there:
    MODE is the safety gate, and every move is announced a few seconds before it starts.

RUN FROM A TERMINAL (laptop on the venue network, or the station):
    pip install viam-sdk numpy            # opencv-python-headless is optional, adds an annotated image
    export VIAM_API_KEY_ID=... VIAM_API_KEY=...     # machine page -> CONNECT -> API keys
    python3 pick_bolt.py --mode look | hover | pick    # asks before every move; --yes to skip
    python3 pick_bolt.py --mode watch                  # continuous; TERMINAL ONLY, see below

What each MODE does:
    look   no motion. Looks from wherever the arm is, prints what it sees and what it would pick.
    hover  home -> look -> close-up look -> open gripper -> stop 90 mm above the bolt.
    pick   the same, then straight down, grab, lift 120 mm. The gripper stays closed.
    box    home -> find the box -> look into it -> print the drop point. Nothing is picked.
    sort   find the box, then repeat: pick the biggest bolt, carry it over the box, drop it, look again,
           until no bolt is left (or MAX_PICKS / the time budget is reached).
    watch  find the box once, then run until Ctrl-C: whenever a bolt or nut is put on the table, pick it and
           drop it in the box. An empty table never ends it. MUST RUN FROM A TERMINAL: the app sandbox stops
           a run at 300 s (mid-move if it comes to that, and the DATA recording is then left running).
           The arm starts only after two frames in a row agree on where the parts are and show no hand (a part
           that shows in only one of the two frames is ignored unless SETTLE_STRICT is set), and
           every part gets a close-up look with the same hand check before the gripper goes down.
           The hand check compares against a clean scan taken at the start (about 10 s): keep hands out for it.
           Put parts fully inside the camera's view (the area is printed at the start): a part cut off by
           the edge of the view, or two parts touching, read as a hand and the arm waits.

Nothing about the parts or the box is stored: every run measures them again with the wrist camera.

Safety: hands, laptops and cables are NOT obstacles the planner knows about; only the configured table,
walls and ceiling are. The arm has collision detection off. Clear the workspace and keep a hand on the
E-stop. Stopping this script does not stop a move that is already running: use the machine's Stop
button or the E-stop.
"""
# ====================================================================================================
# UPSTREAM REFERENCE DEFAULTS. Active values are loaded from config/pick_bolt_controller.yaml below.
# Do not tune these copies: edit the config file instead.
# ====================================================================================================
MODE = "look"              # "look" | "hover" | "pick" | "box" | "sort" | "watch" (watch: terminal only)
PICK_KINDS = ("bolt-standing", "bolt-lying", "nut", "object")   # what counts as pickable; biggest part goes first.
                           # "object": anything else raised that the jaws can take. Remove it to pick fasteners only
CLOSE_UP = "auto"          # "auto": long lying bolts are grasped straight from the scan (faster); nuts and standing
                           # bolts still get the close-up look. "always": close-up for everything
PICK_INDEX = None          # blob number from the look table to go for; None = choose automatically
JAW_AXIS = "y"             # gripper-frame axis the jaws close along. If hover shows them along the bolt, use "x"
TIP_OFFSET_MM = 67.0       # gripper frame origin -> fingertip end (xArm Gripper: 172 mm from flange, frame at 105)
TIP_CLEARANCE_MM = 2.0     # fingertip height above the table when grasping
PAUSE_S = 0.250              # with no terminal to ask: seconds between announcing a move and starting it
GO_HOME_FIRST = True       # hover/pick start with a joint move to the station's home pose
BOX_HINT_XY = None         # e.g. (430, 250): where to look for the box if it cannot be seen from home
# --- where parts go. decide() below returns "box", "reject" or "hold" for every part.
GOOD_XY = (300, 250)       # table spot (x, y) in world mm where GOOD parts are set down, e.g. (300, 250). With it,
                           # sort / watch need no box and nothing is dropped. None = good parts go into the box
REJECT_XY = None           # table spot (x, y) in world mm where rejected parts are set down, e.g. (250, -300)
HOLD_XY = None             # same, for parts the decision is unsure about
ANOMALY_SERVICE = None     # name of the vision service running the anomaly model, e.g. "anomaly".
                           # None = stub: every part is "box"
ANOMALY_REJECT, ANOMALY_HOLD = 0.5, 0.3    # anomaly score >= REJECT -> reject; >= HOLD -> hold; below -> box
LOG_FILE = "pick_bolt_log.csv"             # one line per item, also printed as ITEM lines
MAX_PICKS = 6              # sort mode stops after this many parts
TIME_BUDGET_S = 230        # sort mode starts no new part after this (the app sandbox kills a run at 300 s)
RECORD_HZ = 2.0            # while the program runs, record both cameras + arm + gripper to the Viam DATA tab
                           # at this rate (tagged pick_bolt and run-<time>); 0 = do not record
# --- watch mode
WATCH_PERIOD_S = 1.0       # one frame about this often while nothing is going on
WATCH_STATUS_S = 10.0      # one status line this often
SETTLE_GAP_S, SETTLE_TOL_MM = 0.3, 10.0    # the arm starts after two consecutive frames, at least this far apart,
                                           # that agree on the pickable parts to this tolerance and show no hand
SETTLE_STRICT = False      # True: the two frames must show exactly the same set of parts. False: a part seen in only
                           # one of them is ignored (single depth frames drop thin parts and add ghost bumps; measured
                           # on the arm 22:45 UTC: strict was satisfied in 21 % of frames, this in 60-84 %)
STATIC_FRAMES = 8          # the clean scan is the fused look plus this many single frames, all OR-ed together, so
                           # things that only show in some frames (box edge, cable) are still known as not-a-hand
HAND_OUTSIDE = 0.3         # hand rule: an edge / big / tall blob with more than this share of its pixels outside
STATIC_GROW_PX = 10        # ... the clean scan's raised mask grown by this many pixels is a hand
CLOSE_HAND_WAIT_S = 15.0   # hand seen at the close-up: hold still at most this long, then back up to the scan pose
SKIP_RADIUS_MM, SKIP_TTL_S = 40.0, 60.0    # a skipped spot is left alone until it is empty, or for this long
WATCH_MAX_ERRORS = 3       # errors in a row before watch mode gives up
WATCH_MAX_PICKS, WATCH_MAX_S = 0, 0        # optional caps for watch mode (parts delivered / seconds); 0 = none

# Station constants (read from armfarm22 on 2026-11-13)
DEFAULT_ADDRESS = "armfarm22.310sld03v2.viam.cloud"
ARM, GRIPPER, CAM, MOTION, TELEOP, CAPTURE = "arm", "gripper", "cam", "builtin", "teleop", "capture-control"
HOME_JOINTS_DEG = [0.0, -22.797, -89.785, 0.0, 113.38, 0.0]   # the station's `home-pose`
NATURAL_THETA = 180.0      # gripper theta when pointing straight down at home
TABLE_Z_EXPECTED = 0.0     # the arm base sits on the table. (The configured `table` obstacle top is at -23,
                           # so the planner does NOT protect the last 23 mm above the real table.)
TABLE_Z_RANGE = (-20.0, 30.0)   # measured table height must fall in here; depth read +17..21 from 620 mm, +3 from 470
TABLE_READS_HIGH = {"close": 2.5, "scan": 5.0}   # mm the wrist depth over-reads the table at 340 / 440 mm (measured here)
LOW_PART_MM, LOW_PART_TIP_MM = 15.0, 2.5         # nuts and other low parts: fingertips this close to the table
THIN_PART_MM, THIN_PART_TIP_MM = 7.0, 1.5        # thin (jam) nuts, about 5 mm tall: lower still
DARK_MIN = 25.0                                  # color pass: a patch this much darker than the table around it
SCAN_HEIGHT_MM = 440.0     # camera height for the scan. From home (620 mm) the depth noise swamps 10 mm objects
FRAMES = 4                 # point clouds fused per scan look (per-pixel median): single frames show ghost bumps
CLOSE_FRAMES = 3           # ... and per close-up look, where the depth is much cleaner
PAD_LEN_MM = 25.0          # usable finger pad length above the fingertip end
APPROACH_MM, LIFT_MM, REFINE_HEIGHT_MM = 60.0, 120.0, 340.0
MAX_NEIGHBOUR_MM = 45.0    # anything taller than this where the gripper body would go blocks the pick
BOX_MIN_HEIGHT_MM = 80.0   # a box candidate is at least this tall ...
BOX_MIN_AREA_MM2 = 6000.0  # ... and shows at least this much of itself
BOX_LOOK_INSET_MM = 140.0  # first look into the box: this far past its nearest rim
BOX_LOOK_CAM_Z = 600.0     # camera height for that look (fingertips at 446: clears a 400 mm box)
WALL_MARGIN_MM = 70.0      # the drop point stays this far from the box walls
DROP_CLEAR_MM = 90.0       # fingertip height over the rim while carrying over the box
# Letting go: the part is lowered into the box, not dropped from carrying height. How far below the rim the
# fingertips go depends on the room between the drop point and the walls (release_tip_z).
DROP_ABOVE_FLOOR_MM = 60.0 # ... but never closer to the box floor than this: the part hangs below the fingertips
                           # and the pile grows. Set it very high (1000) to drop from carrying height as before
FINGERS_IN_BOX_MM = 45.0   # below the rim when only the fingers fit (they are ~60 mm deep): needs JAW_CLEAR_MM of room
BODY_IN_BOX_MM = 120.0     # below the rim when the gripper body fits too: needs BODY_CLEAR_MM of room, so raise
                           # WALL_MARGIN_MM to that for a deep box. The wrist camera, 154 mm above the fingertips, stays out
JAW_CLEAR_MM = 60.0        # room the open jaws need from the box walls ...
RIM_HOVER_MM = 10.0        # ... with less, the fingertips stop this far above the rim instead
REACH_MM, SHOULDER_Z, WRIST_ABOVE_GRIPPER = 615.0, 267.0, 202.0   # xArm6: wrist centre must stay inside this sphere
WORKSPACE = dict(x=(150.0, 620.0), y=(-380.0, 400.0))   # world mm; walls are at x=690 and y=450

# Size gates, mm. Loosen these if your bolts differ.
RAISED_MIN_MM = 5.0        # a blob must rise this far above the table to count as a 3D object (nuts read 7-14)
FLAT_MAX_MM = 3.5          # below this it is a washer, a mark or a shadow
# Measured on this table: cap bolt lying 83 x 31 x 22, lag screw lying 69 x 21 x 12, flange bolt standing
# 37 x 29 x 41, nuts up to 27 x 25 x 14.
BOLT_STANDING_MIN_H, BOLT_MAX_H = 18.0, 130.0    # the cap bolt on its head stands 83 mm; taller is not a bolt
BOLT_STANDING_MAX_LEN = 90.0   # depth sometimes shows a standing bolt as a low mound up to ~90 mm across
STANDING_TIP_MM = 27.0         # fingertip height for every standing bolt; its measured height is NOT used (depth
                               # is wrong both ways on thin standing parts). 23..31 mm suits bolts 37..87 mm tall:
                               # >= 10 mm of the 25 mm pad on a 41 mm bolt, and the gripper body (fingers ~60 mm
                               # deep - measure yours) clears an 83 mm bolt by about 4 mm
BODY_CLEAR_MM, CAM_CLEAR_MM = 80.0, 65.0   # room the gripper body / the wrist camera need from the box walls
SPOT_KEEPOUT_MM = 150.0        # reject / hold spots must be this far from the box
PLACE_PITCH_MM = 35.0          # a spot is a grid of places this far apart: PLACE_COLS along +x, PLACE_ROWS away from
PLACE_COLS, PLACE_ROWS = 3, 4  # the parts (+y on the +y side of the arm, -y on the -y side). A part takes the first
PLACE_GAP_MM = 8.0             # place where the camera sees this much free table around it, and nothing within reach of
JAW_SWEEP_MM, JAW_PAD_MM = 55.0, 20.0   # the opening jaws: they open across the part's middle (+-JAW_PAD_MM along
                               # it) out to JAW_SWEEP_MM from its centre line (the xArm Gripper opens about 85 mm, plus
                               # the fingers). No free place: the part is put back where it was picked
PLACE_CARRY_MIN_Z = 200.0      # a place out of reach at carrying height is approached lower, never below this gripper
                               # height; every place must be reachable here
TAPE_MAX_MM = 2.0              # blobs this flat (tape marking a spot, a cable) do not take a place; anything taller does
# A standing bolt is aimed at by its top, seen from the scan height (from the close-up, 340 mm up, the top of an 80 mm
# bolt is nearer than the depth camera's minimum range: it read 24 x 8 x 18 mm and the jaws closed on air beside it).
# summit(): the highest SUMMIT_MIN_PX points within SUMMIT_RADIUS_MM, and those within SUMMIT_BAND_MM below them.
SUMMIT_RADIUS_MM, SUMMIT_BAND_MM, SUMMIT_MIN_PX = 35.0, 6.0, 12
STANDING_MIN_TOP_MM = 40.0     # a top this high is a standing part; lying parts here read 30 mm at most
STANDING_AIM_MM = 25.0         # an external target is standing only if that top is this near its position
STANDING_OWN_MM = 50.0         # depth smear this near a standing part's top is the part itself, not a tall neighbour
TOP_CLEAR_MM = 25.0            # the open fingertips arrive at least this far above a standing part's top
STANDING_BODY_CLEAR_MM = 83.0  # tallest standing part the gripper body clears with fingertips at STANDING_TIP_MM
BOLT_LYING_MIN_H, BOLT_LYING_MAX_H = 6.0, 40.0
BOLT_LYING_MIN_LEN, BOLT_LYING_MIN_ASPECT, BOLT_LYING_MAX_W = 25.0, 1.7, 45.0
GRIP_MAX_SPAN_MM = 70.0        # widest thing the open jaws take with room to spare (the xArm Gripper opens about
                               # 85 mm - measure yours). grasp_plan() refuses anything wider
GRIP_MIN_H_MM = 5.5            # lower than this there is nothing for the pads to hold
MAX_OBJECT_LEN, MIN_AREA_MM2, MAX_AREA_MM2 = 110.0, 20.0, 4000.0
NUT_MAX_LEN, NUT_MIN_AREA = 45.0, 35.0     # was 32 / 50: a nut with a tape tag measured 33 x 24 and flipped to
                                           # "unknown"; a small plain nut measured 9 x 8 (run of 19:08 UTC)
# ====================================================================================================

import argparse
import asyncio
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np  # Dependencies must be installed explicitly; never auto-install at import.
try:
    if os.environ.get("PICK_BOLT_NO_CV2"):
        raise ImportError
    import cv2   # optional: adds flat/no-depth blobs and an annotated image
except ImportError:
    cv2 = None

# Preserve the upstream references above, but configuration owns all active settings.
from pathlib import Path
import yaml
_controller_keys = {name for name in globals() if name.isupper()}
_controller_settings = yaml.safe_load((Path(__file__).resolve().parent / 'config/pick_bolt_controller.yaml').read_text())
if set(_controller_settings) != _controller_keys:
    raise ValueError('Controller configuration keys must match the imported controller settings')
globals().update(_controller_settings)


# ----------------------------------------------------------------------------------------------------
# Perception (numpy only; no robot needed, testable offline)
# ----------------------------------------------------------------------------------------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class Obj:
    idx: int
    kind: str                 # bolt-standing | bolt-lying | bolt-lying? | nut | object | flat | dark | round? | unknown | rejected:<why>
    score: float              # 0..1, how sure this is a bolt
    u: float
    v: float
    length_mm: float
    width_mm: float
    height_mm: Optional[float]
    area_mm2: float
    p_table: np.ndarray       # camera frame: point on the table under the blob centroid
    p_axis: np.ndarray        # camera frame: second table point along the blob's long axis
    ys: np.ndarray = field(repr=False, default=None)
    xs: np.ndarray = field(repr=False, default=None)
    source: str = "depth"     # "depth": rose above the table; "dark": seen by color, too low for the depth cut


def _pcd_header(data: bytes) -> Tuple[dict, int]:
    header, pos = {}, 0
    while True:
        end = data.index(b"\n", pos)
        line = data[pos:end].decode("ascii", "replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition(" ")
        header[key.upper()] = val.split()
        if key.upper() == "DATA":
            return header, pos


def pcd_summary(data: bytes) -> str:
    h, _ = _pcd_header(data)
    return (f"{h['POINTS'][0]} points, fields {' '.join(h['FIELDS'])}, "
            f"{h.get('WIDTH', ['?'])[0]} x {h.get('HEIGHT', ['?'])[0]}, {h['DATA'][0]}")


def parse_pcd(data: bytes, want_rgb: bool = False):
    """PCD bytes -> Nx3 float array in mm, camera frame. Handles binary and ascii, metres or mm.
    With want_rgb: also an N-vector of gray levels from the cloud's own colors (None if it has none)."""
    header, pos = _pcd_header(data)
    fields, sizes = header["FIELDS"], [int(s) for s in header["SIZE"]]
    types, counts = header["TYPE"], [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    n = int(header["POINTS"][0])
    mode = header["DATA"][0].lower()
    packed = None
    if mode == "binary":
        kind = {"F": "f", "I": "i", "U": "u"}
        dtype = np.dtype([(f, "<" + kind[t] + str(s), (c,)) if c > 1 else (f, "<" + kind[t] + str(s))
                          for f, s, t, c in zip(fields, sizes, types, counts)])
        arr = np.frombuffer(data, dtype=dtype, count=n, offset=pos)
        pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        if "rgb" in fields and arr["rgb"].dtype.itemsize == 4:
            packed = np.ascontiguousarray(arr["rgb"]).view("<u4")      # 0x00RRGGBB, also when stored as float
    elif mode == "ascii":
        raw = np.loadtxt(data[pos:].decode("ascii").splitlines(), ndmin=2)
        pts = raw[:, [fields.index("x"), fields.index("y"), fields.index("z")]].astype(np.float64)
        if "rgb" in fields:
            packed = raw[:, fields.index("rgb")].astype(np.uint32)
    else:
        raise ValueError(f"unsupported PCD encoding: {mode}")
    ok = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts = pts[ok]
    if len(pts) and np.median(pts[:, 2]) < 20.0:   # metres -> mm
        pts = pts * 1000.0
    if not want_rgb:
        return pts
    gray = None
    if packed is not None:
        packed = packed[ok]
        gray = (0.299 * ((packed >> 16) & 255) + 0.587 * ((packed >> 8) & 255) + 0.114 * (packed & 255)).astype(np.float32)
    return pts, gray


def depth_from_points(pts: np.ndarray, K: Intrinsics, values: Optional[np.ndarray] = None):
    """Project camera-frame points into the color image: HxW depth in mm, 0 where there is no data.
    With values (one per point): also an HxW image of those values, NaN where there is no data."""
    z = pts[:, 2]
    u = np.rint(K.fx * pts[:, 0] / z + K.cx).astype(int)
    v = np.rint(K.fy * pts[:, 1] / z + K.cy).astype(int)
    ok = (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
    depth = np.zeros((K.height, K.width), np.float32)
    order = np.argsort(-z[ok])                      # far first, so the nearest point wins a pixel
    depth[v[ok][order], u[ok][order]] = z[ok][order]
    if values is None:
        return depth
    img = np.full((K.height, K.width), np.nan, np.float32)
    img[v[ok][order], u[ok][order]] = values[ok][order]
    return depth, img


def fit_table_plane(P: np.ndarray, valid: np.ndarray, tol_mm: float = 4.0) -> Tuple[np.ndarray, float]:
    """RANSAC plane through the camera-frame point image P. Returns (n, d) with height(p) = n.p + d,
    positive toward the camera."""
    pts = P[::4, ::4][valid[::4, ::4]]
    if len(pts) < 200:
        raise RuntimeError("too few depth points to find the table")
    rng = np.random.default_rng(0)
    best, best_n, best_d = -1, None, 0.0
    for _ in range(300):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        count = int((np.abs((pts - a) @ n) < tol_mm).sum())
        if count > best:
            best, best_n, best_d = count, n, -float(a @ n)
    inl = pts[np.abs(pts @ best_n + best_d) < tol_mm]
    c = inl.mean(axis=0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][2]
    d = -float(c @ n)
    if d < 0:                                       # camera origin must be on the positive side
        n, d = -n, -d
    return n, d


def _hood(mask: np.ndarray) -> list:
    """The nine 3x3-neighbourhood shifts of a boolean image."""
    H, W = mask.shape
    p = np.pad(mask, 1)
    return [p[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)]


def dilate(mask: np.ndarray, times: int = 1) -> np.ndarray:
    for _ in range(times):
        mask = np.logical_or.reduce(_hood(mask))
    return mask


def opened(mask: np.ndarray) -> np.ndarray:
    """3x3 morphological opening: removes specks, keeps blobs."""
    return dilate(np.logical_and.reduce(_hood(mask)))


def components(mask: np.ndarray, min_px: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
    """8-connected components of a boolean image -> [(ys, xs)], pure Python flood fill."""
    H, W = mask.shape
    rows = [bytearray(r) for r in mask.astype(np.uint8)]
    out = []
    for y0, x0 in zip(*(a.tolist() for a in np.nonzero(mask))):
        if not rows[y0][x0]:
            continue
        rows[y0][x0] = 0
        stack, ys, xs = [(y0, x0)], [], []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for yy in (y - 1, y, y + 1):
                if 0 <= yy < H:
                    row = rows[yy]
                    for xx in (x - 1, x, x + 1):
                        if 0 <= xx < W and row[xx]:
                            row[xx] = 0
                            stack.append((yy, xx))
        if len(ys) >= min_px:
            out.append((np.array(ys), np.array(xs)))
    return out


def fuse_depth(frames: List[np.ndarray]) -> np.ndarray:
    """Per-pixel median of several depth images (0 = no data). A pixel needs data in most frames."""
    import warnings
    stack = np.stack(frames).astype(np.float32)
    stack[stack <= 0] = np.nan
    enough = np.isfinite(stack).sum(axis=0) > len(frames) // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # pixels with no depth in any frame
        med = np.nanmedian(stack, axis=0)
    return np.where(enough, np.nan_to_num(med, nan=0.0), 0.0).astype(np.float32)


def local_background(img: np.ndarray, block: int = 80) -> np.ndarray:
    """Table brightness everywhere: median of each block x block cell, blended smoothly. NaN = no data."""
    import warnings
    H, W = img.shape
    gy, gx = H // block, W // block
    cells = img[:gy * block, :gx * block].reshape(gy, block, gx, block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(cells, axis=(1, 3))
        fill = np.nanmedian(img)
    med = np.where(np.isfinite(med), med, fill)
    ys, xs = (np.arange(H) + 0.5) / block - 0.5, (np.arange(W) + 0.5) / block - 0.5
    y0, x0 = np.clip(np.floor(ys).astype(int), 0, gy - 1), np.clip(np.floor(xs).astype(int), 0, gx - 1)
    y1, x1 = np.clip(y0 + 1, 0, gy - 1), np.clip(x0 + 1, 0, gx - 1)
    fy, fx_ = np.clip(ys - y0, 0, 1)[:, None], np.clip(xs - x0, 0, 1)[None, :]
    top = med[y0][:, x0] * (1 - fx_) + med[y0][:, x1] * fx_
    bot = med[y1][:, x0] * (1 - fx_) + med[y1][:, x1] * fx_
    return top * (1 - fy) + bot * fy


def detect(pts: np.ndarray, K: Intrinsics, bgr: Optional[np.ndarray] = None,
           cloud_gray: Optional[np.ndarray] = None) -> Tuple[List[Obj], dict]:
    """Single-cloud convenience wrapper around detect_depth."""
    if cloud_gray is not None:
        depth, gray_img = depth_from_points(pts, K, cloud_gray)
    else:
        depth, gray_img = depth_from_points(pts, K), None
    objs, info = detect_depth(depth, K, bgr, gray_img)
    info["in_frame"] = float((depth > 0).sum() / max(len(pts), 1))
    return objs, info


def detect_depth(depth: np.ndarray, K: Intrinsics, bgr: Optional[np.ndarray] = None,
                 gray_img: Optional[np.ndarray] = None) -> Tuple[List[Obj], dict]:
    """Find objects on the table from a depth image in the color camera's pixel grid."""
    H, W = K.height, K.width
    valid = depth > 0
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.dstack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones((H, W))])
    P = rays * depth[..., None]
    n, d = fit_table_plane(P, valid)
    height = np.where(valid, P @ n + d, np.nan)
    # Noise of the table surface itself sets how high a blob must rise to be believed.
    flat_h = height[valid & (np.abs(np.nan_to_num(height, nan=99.0)) < RAISED_MIN_MM)]
    noise = float(1.4826 * np.median(np.abs(flat_h - np.median(flat_h)))) if flat_h.size else 0.0
    raised_min = max(RAISED_MIN_MM, 4.0 * noise)

    # Pass A, depth: blobs that rise above the table. A cable lying next to a bolt stays separate.
    raised = opened(np.nan_to_num(height, nan=0.0) >= raised_min)
    passes = [("depth", raised)]
    # A lower cut shows the thin parts too (a screw shank next to its head); used to find mid-length.
    skirt_label = np.zeros((H, W), np.int32)
    skirts = components(opened(np.nan_to_num(height, nan=0.0) >= max(3.0, 3.0 * noise)))
    for i, (sy, sx) in enumerate(skirts, 1):
        skirt_label[sy, sx] = i
    # Pass B, color (needs OpenCV): what differs from the local table color and shows no height
    # (washers, marks) or sits where the depth has holes (shiny metal).
    if gray_img is not None:
        # Pass C, the cloud's own colors: a thin nut reads only 3-4 mm in depth at close range (seen on the arm)
        # but is far darker than the table. Shape from color, height from whatever depth it does have.
        dark = np.nan_to_num(local_background(gray_img) - gray_img, nan=0.0) >= DARK_MIN
        passes.append(("dark", opened(dark & ~dilate(raised, 3))))
    elif cv2 is not None and bgr is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
        color = (cv2.absdiff(gray, cv2.medianBlur(gray, 51)) > 25) | \
                (sat.astype(int) - cv2.medianBlur(sat, 51).astype(int) > 40)
        passes.append(("color", opened(color & ~dilate(raised, 2))))

    def on_plane(u, v, h):
        """Camera-frame table point under the point at height h seen through pixel (u, v)."""
        r = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
        p = r * ((h - d) / float(n @ r))
        return p - n * h, p[2]

    objs: List[Obj] = []
    for source, mask in passes:
        for ys, xs in components(mask):
            hs_all = height[ys, xs]
            hs = hs_all[np.isfinite(hs_all)]
            cys, cxs, h_ray = ys, xs, 0.0
            if source == "depth":
                obj_h = float(np.percentile(hs, 95))
                # Stereo depth smears small tall things and thins low ones at the cut-off. Measure the
                # part above half height instead: that is the object, not its skirt.
                core = np.nan_to_num(hs_all, nan=0.0) >= max(raised_min, 0.5 * obj_h)
                if core.sum() >= 12:
                    cys, cxs = ys[core], xs[core]
                h_ray = float(np.nanmean(height[cys, cxs]))       # height where the rays through the core land
            elif source == "dark":   # low part: its highest decile is the best the depth can say
                obj_h = float(max(0.0, np.percentile(hs, 90))) if hs.size >= max(12, 0.3 * len(ys)) else None
                h_ray = (obj_h or 0.0) / 2
            else:   # height known only if most of the blob has depth
                obj_h = float(max(0.0, np.percentile(hs, 95))) if hs.size >= max(12, 0.5 * len(ys)) else None
                h_ray = (obj_h or 0.0) / 2
            cu, cv_ = cxs.mean(), cys.mean()
            # Principal axis in pixels.
            major = np.linalg.eigh(np.cov(np.stack([cxs - cu, cys - cv_])))[1][:, 1]
            along = (cxs - cu) * major[0] + (cys - cv_) * major[1]
            across = (cxs - cu) * -major[1] + (cys - cv_) * major[0]
            len_px, wid_px = np.ptp(along) + 1, np.ptp(across) + 1
            sid = int(skirt_label[ys[0], xs[0]]) if source == "depth" else 0
            if sid and len_px / max(wid_px, 1e-6) >= 1.4 and len(skirts[sid - 1][0]) <= 6 * len(ys):
                # Long part: the tall end (head) pulls the centroid toward itself. Use the middle of the
                # whole length instead, so the part hangs level in the jaws.
                sy, sx = skirts[sid - 1]
                s_along = (sx - cu) * major[0] + (sy - cv_) * major[1]
                s_across = (sx - cu) * -major[1] + (sy - cv_) * major[0]
                s_along = s_along[np.abs(s_across) <= wid_px]
                if s_along.size >= 12:
                    lo, hi = np.percentile(s_along, [2, 98])
                    cu, cv_ = cu + 0.5 * (lo + hi) * major[0], cv_ + 0.5 * (lo + hi) * major[1]
                    len_px = max(len_px, hi - lo + 1)
            p_table, z_obj = on_plane(cu, cv_, h_ray)
            p_axis, _ = on_plane(cu + 25 * major[0], cv_ + 25 * major[1], h_ray)
            mm = z_obj / K.fx
            L, Wd, area = len_px * mm, wid_px * mm, len(cys) * mm * mm
            aspect = L / max(Wd, 1e-6)

            if xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3:
                kind, score = "rejected:edge", 0.0
            elif area < MIN_AREA_MM2:
                kind, score = "rejected:tiny", 0.0
            elif L > MAX_OBJECT_LEN or area > MAX_AREA_MM2:
                kind, score = "rejected:big", 0.0
            elif source == "dark":
                # Seen by color only. A washer and a thin nut look alike here (2 mm vs 5 mm is inside the depth
                # noise), so this never starts a pick. It only confirms, at the close-up, a part the scan measured.
                kind, score = "dark", 0.0
            elif obj_h is None:                       # no depth on it: shape is all we have
                if L >= BOLT_LYING_MIN_LEN and aspect >= BOLT_LYING_MIN_ASPECT and Wd <= BOLT_LYING_MAX_W:
                    kind, score = "bolt-lying?", 0.5
                elif L <= 30:
                    kind, score = "round?", 0.3       # standing bolt or nut, cannot tell
                else:
                    kind, score = "unknown", 0.1
            elif obj_h < FLAT_MAX_MM:
                kind, score = "flat", 0.0             # washer, sticker, shadow
            elif obj_h > BOLT_MAX_H:
                kind, score = "rejected:tall", 0.0
            elif BOLT_LYING_MIN_H <= obj_h <= BOLT_LYING_MAX_H and L >= BOLT_LYING_MIN_LEN \
                    and aspect >= BOLT_LYING_MIN_ASPECT and Wd <= BOLT_LYING_MAX_W:
                kind, score = "bolt-lying", 0.8
            elif obj_h >= BOLT_STANDING_MIN_H and L <= BOLT_STANDING_MAX_LEN and area >= 40.0:
                kind, score = "bolt-standing", (0.9 if L <= 45 else 0.7)
            elif L <= NUT_MAX_LEN and area >= NUT_MIN_AREA and obj_h >= GRIP_MIN_H_MM:
                kind, score = "nut", 0.6
            elif obj_h >= GRIP_MIN_H_MM and area >= 40.0 and grasp_plan(L, Wd, obj_h) is not None:
                kind, score = "object", 0.5           # not bolt- or nut-shaped, but raised and the jaws can take it
            else:
                kind, score = "unknown", 0.1
            objs.append(Obj(len(objs), kind, score, float(cu), float(cv_), float(L), float(Wd), obj_h, float(area),
                            p_table, p_axis, ys, xs, source))

    info = dict(height=height, raised=raised, valid=valid, depth=depth, cam_height_mm=d, gray=gray_img, foot=-n * d,
                noise_mm=noise, raised_min=raised_min,
                tilt_deg=math.degrees(math.acos(min(1.0, abs(n[2])))), depth_coverage=float(valid.mean()),
                in_frame=1.0, color_pass=len(passes) > 1)

    return objs, info


def choose_bolt(objs: List[Obj], K: Intrinsics, index: Optional[int] = None) -> Optional[Obj]:
    if index is not None:
        return next((o for o in objs if o.idx == index), None)
    bolts = [o for o in objs if o.kind in PICK_KINDS]   # depth-confirmed only
    # The biggest one: easiest to see and to hold. PICK_INDEX overrides.
    return max(bolts, key=lambda o: o.area_mm2, default=None)


def neutral_theta(x: float, y: float) -> float:
    """Gripper theta that keeps the wrist camera pointing away from the arm's base at world (x, y).
    The camera sits 83 mm out along the gripper x axis, (-cos t, sin t, 0). Turned more than about 100
    degrees from 'outward' it runs into the forearm, and the planner refuses the move (seen on the arm:
    theta -113 at azimuth 40 deg failed)."""
    return (360.0 - math.degrees(math.atan2(y, x))) % 360.0 - 180.0


def camera_off_outward_deg(theta: float, x: float, y: float) -> float:
    """How far the camera points from radially outward, 0..180."""
    return abs((180.0 - theta - math.degrees(math.atan2(y, x)) + 180.0) % 360.0 - 180.0)


def grasp_theta(long_axis_world_deg: float, jaw_axis: str = "y", near: float = NATURAL_THETA) -> float:
    """Gripper theta (pointing straight down) that closes the jaws across a bolt lying along the given
    world direction; of the two ways round, the one nearest `near`. Verified on the station: at theta t
    the gripper y axis is (sin t, cos t, 0) and the x axis is (-cos t, sin t, 0)."""
    t = -long_axis_world_deg if jaw_axis == "y" else 90.0 - long_axis_world_deg
    best = min((t + k * 180.0 for k in range(-4, 5)), key=lambda c: abs(c - near))
    return (best + 180.0) % 360.0 - 180.0


def grasp_plan(length_mm: float, width_mm: float, height_mm: Optional[float], upright: bool = False,
               tip_clearance_mm: float = TIP_CLEARANCE_MM) -> Optional[Tuple[bool, bool, float]]:
    """How to take a part, from its measured size alone: (across, upright, fingertip height above the table),
    or None if the open jaws cannot take it. What the part is called plays no role.
    across:  the jaws must close across the long axis (a long part, or one too long to take the other way).
    upright: tall and not long. The depth is wrong both ways on these, in height and in footprint, so the
             fingertip height is fixed and the footprint is not held against the jaw opening. Pass upright=True
             when an earlier look already said so: it sticks."""
    h = height_mm or 0.0
    aspect = length_mm / max(width_mm, 1e-6)
    long_low = (BOLT_LYING_MIN_H <= h <= BOLT_LYING_MAX_H and length_mm >= BOLT_LYING_MIN_LEN
                and aspect >= BOLT_LYING_MIN_ASPECT and width_mm <= BOLT_LYING_MAX_W)
    if upright:
        return False, True, STANDING_TIP_MM
    if h >= BOLT_STANDING_MIN_H and not long_low:
        return (False, True, STANDING_TIP_MM) if length_mm <= BOLT_STANDING_MAX_LEN else None
    across = aspect >= BOLT_LYING_MIN_ASPECT or length_mm > GRIP_MAX_SPAN_MM
    if (width_mm if across else length_mm) > GRIP_MAX_SPAN_MM:
        return None
    tip = max(tip_clearance_mm, h - PAD_LEN_MM)
    if 0 < h <= LOW_PART_MM:      # a nut is only 7-10 mm tall: every mm of fingertip height is grip lost
        tip = min(tip, LOW_PART_TIP_MM if h >= THIN_PART_MM else THIN_PART_TIP_MM)
    return across, False, tip


def report(objs: List[Obj], info: dict, K: Intrinsics) -> str:
    """Table of blobs plus a text picture of the camera view, for consoles that cannot show images."""
    lines = [f"  camera {info['cam_height_mm']:.0f} mm from the table, tilt {info['tilt_deg']:.1f} deg, "
             f"depth coverage {info['depth_coverage'] * 100:.0f}%, table noise {info['noise_mm']:.1f} mm, "
             f"blobs must rise {info['raised_min']:.0f} mm"
             + ("" if info["color_pass"] else "  (no OpenCV: flat and no-depth blobs are not listed)"),
             "  idx kind            len x wid x height mm    pixel"]
    for o in objs:
        hs = "  ?" if o.height_mm is None else f"{o.height_mm:3.0f}"
        lines.append(f"  {o.idx:3d} {o.kind:15s} {o.length_mm:4.0f} x {o.width_mm:3.0f} x {hs}        ({o.u:.0f},{o.v:.0f})")
    # 80 x 30 character map: '.' table, ' ' no depth, '#' raised, numbers mark the blobs worth a look.
    cw, ch = K.width // 80, K.height // 30
    crop = (slice(0, ch * 30), slice(0, cw * 80))
    cells = lambda a: a[crop].reshape(30, ch, 80, cw)                       # noqa: E731
    grid = np.where(cells(info["raised"]).any(axis=(1, 3)), "#",
                    np.where(cells(info["valid"]).mean(axis=(1, 3)) < 0.5, " ", "."))
    rows = [list(r) for r in grid]
    for o in objs:
        if not o.kind.startswith("rejected"):
            r, c = min(29, int(o.v) // ch), min(79, int(o.u) // cw)
            for i, char in enumerate(str(o.idx)):
                if c + i < 80:
                    rows[r][c + i] = char
    lines.append("  view ('.' table, '#' raised, blank no depth, numbers = blob idx):")
    lines += ["  |" + "".join(r) + "|" for r in rows]
    return "\n".join(lines)


def zoom_map(info: dict, u: float, v: float, cols: int = 56, rows: int = 28) -> str:
    """Close-up of the height image around pixel (u, v), 2 x 4 pixels per character:
    '.' table, ':' 2-5 mm, digit n = n x 5 mm or more, '+' over 50 mm, blank = no depth.
    If the cloud carries colors, a second picture marks with '#' where it is darker than the table."""
    height, gray = info["height"], info.get("gray")
    H, W = height.shape
    x0 = int(min(max(0, u - cols), W - 2 * cols))
    y0 = int(min(max(0, v - 2 * rows), H - 4 * rows))
    cell = lambda a: a[y0:y0 + 4 * rows, x0:x0 + 2 * cols].reshape(rows, 4, cols, 2)   # noqa: E731
    hc = cell(height)
    has = np.isfinite(hc).any(axis=(1, 3))
    hmax = np.nanmax(np.where(np.isfinite(hc), hc, -1e9), axis=(1, 3))
    chars = np.where(~has, " ", np.where(hmax < 2, ".", np.where(hmax < 5, ":", np.where(hmax > 50, "+", "0"))))
    digits = np.clip((hmax // 5).astype(int), 1, 9).astype(str)
    left = np.where(chars == "0", digits, chars)
    out = [f"  close-up at pixel ({u:.0f},{v:.0f}), columns {x0}..{x0 + 2 * cols}, rows {y0}..{y0 + 4 * rows}; "
           "'.' table ':' 2-5 mm, digit n = n x 5 mm, blank no depth" + ("; right: '#' = cloud darker than table" if gray is not None else "")]
    right = None
    if gray is not None:
        gc = cell(gray)
        ghas = np.isfinite(gc).any(axis=(1, 3))
        gmin = np.nanmin(np.where(np.isfinite(gc), gc, 1e9), axis=(1, 3))
        table_gray = float(np.nanmedian(gray))
        right = np.where(~ghas, " ", np.where(gmin < table_gray - 35, "#", "."))
    for r in range(rows):
        out.append("  |" + "".join(left[r]) + "|" + ("  |" + "".join(right[r]) + "|" if right is not None else ""))
    return "\n".join(out)


def annotate(bgr: np.ndarray, objs: List[Obj]) -> np.ndarray:
    vis = bgr.copy()
    for o in objs:
        col = (0, 200, 0) if o.score >= 0.8 else (0, 165, 255) if o.score > 0 else (160, 160, 160)
        cv2.rectangle(vis, (int(o.xs.min()), int(o.ys.min())), (int(o.xs.max()), int(o.ys.max())), col, 1)
        hs = "?" if o.height_mm is None else f"{o.height_mm:.0f}"
        cv2.putText(vis, f"{o.idx}:{o.kind.split(':')[-1]} {o.length_mm:.0f}x{o.width_mm:.0f}x{hs}",
                    (int(o.xs.max()) + 4, int(o.v)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return vis


# ----------------------------------------------------------------------------------------------------
# Box (pure functions on world-frame X, Y, Z images)
# ----------------------------------------------------------------------------------------------------
def world_maps(depth: np.ndarray, K: Intrinsics, R: np.ndarray, t: np.ndarray):
    """Depth image -> world X, Y, Z images (NaN where there is no depth)."""
    uu, vv = np.meshgrid(np.arange(K.width), np.arange(K.height))
    P = np.dstack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones(depth.shape)]) * depth[..., None]
    Pw = P @ R.T + t
    Pw[depth <= 0] = np.nan
    return Pw[..., 0], Pw[..., 1], Pw[..., 2]


def find_tall_object(X, Y, Z, depth, K: Intrinsics, cam_xy) -> Optional[dict]:
    """Biggest thing at least BOX_MIN_HEIGHT_MM tall. Returns its rim height and where to look into it."""
    tall = opened(np.nan_to_num(Z, nan=-1e9) >= TABLE_Z_EXPECTED + BOX_MIN_HEIGHT_MM)
    best = None
    for ys, xs in components(tall, min_px=300):
        area = float(((depth[ys, xs] / K.fx) ** 2).sum())
        if area >= BOX_MIN_AREA_MM2 and (best is None or area > best[0]):
            best = (area, ys, xs)
    if best is None:
        return None
    area, ys, xs = best
    zs = Z[ys, xs]
    rim = min(float(np.percentile(zs, 97)), robust_top(zs))
    band = zs >= 0.7 * rim
    xy = np.stack([X[ys, xs][band], Y[ys, xs][band]], axis=1)
    cam_xy = np.asarray(cam_xy, float)
    near = xy[np.argmin(np.linalg.norm(xy - cam_xy, axis=1))]
    away = xy.mean(axis=0) - cam_xy
    away = away / max(np.linalg.norm(away), 1e-6)
    return dict(rim=rim, area=area, near=near, look_xy=near + away * BOX_LOOK_INSET_MM)


def robust_top(zs: np.ndarray, band_mm: float = 30.0, min_frac: float = 0.03) -> float:
    """Top of the highest well-populated height band. A box rim is thousands of pixels within a few cm of
    height; a cable or a depth spike above it is a few hundred spread over much more, and is ignored."""
    zs = np.sort(np.asarray(zs, float))
    need = max(50, int(min_frac * len(zs)))
    support = np.searchsorted(zs, zs, side="right") - np.searchsorted(zs, zs - band_mm, side="left")
    ok = np.nonzero(support >= need)[0]
    return float(zs[ok[-1]]) if len(ok) else float(np.percentile(zs, 97))


def find_drop_point(X, Y, Z, depth, K: Intrinsics, rim_guess: float) -> dict:
    """Seen from above the box: confirm walls around a floor, and choose where to let go.
    Returns dict(ok, reason, drop, rim, walls)."""
    H, W = Z.shape
    valid = np.isfinite(Z)
    # The far look may have seen only part of a wall, so these cuts lean on absolute heights.
    wall = valid & (Z >= TABLE_Z_EXPECTED + max(60.0, 0.4 * rim_guess))
    floor = valid & (Z <= TABLE_Z_EXPECTED + max(25.0, 0.2 * rim_guess))
    if wall.sum() < 200:
        return dict(ok=False, reason="no rim or wall in view")
    rim = float(np.percentile(Z[wall], 97))
    fy, fx_ = np.nonzero(floor)
    if fy.size < 500:
        return dict(ok=False, reason="no floor in view", rim=rim)
    # A floor pixel is inside the box if walls stand on both sides of it along one image axis and on at
    # least one side along the other. Table outside the box has a wall on one side only.
    acc = np.logical_or.accumulate
    left, right = acc(wall, axis=1), acc(wall[:, ::-1], axis=1)[:, ::-1]
    up, down = acc(wall, axis=0), acc(wall[::-1], axis=0)[::-1]
    count = left.astype(int) + right + up + down
    enclosed = floor & (count >= 3) & ((left & right) | (up & down))
    ey_, ex_ = np.nonzero(enclosed)
    if ey_.size < 500:
        sy, sx = int(fy[np.argmin((fy - K.cy) ** 2 + (fx_ - K.cx) ** 2)]), int(fx_[np.argmin((fy - K.cy) ** 2 + (fx_ - K.cx) ** 2)])
        seen = [k for k, m in (("left", left), ("right", right), ("up", up), ("down", down)) if m[sy, sx]]
        return dict(ok=False, reason=f"walls on {len(seen)} of 4 sides only ({', '.join(seen) or 'none'}): not inside a box",
                    rim=rim, walls=seen)
    # The enclosed floor region nearest the image centre (the camera was aimed into the box).
    i = int(np.argmin((ey_ - K.cy) ** 2 + (ex_ - K.cx) ** 2))
    sy, sx = int(ey_[i]), int(ex_[i])
    interior = None
    for ys, xs in components(enclosed, min_px=500):
        if ((ys == sy) & (xs == sx)).any():
            interior = np.zeros((H, W), bool)
            interior[ys, xs] = True
            break
    if interior is None:
        return dict(ok=False, reason="enclosed floor is too small", rim=rim)
    walls = [k for k, m in (("left", left), ("right", right), ("up", up), ("down", down)) if m[sy, sx]]
    # Rim height from the walls right around that floor only (a partition behind the box must not count).
    iy, ix = np.nonzero(interior)
    # Seen from above, the rim sits outside the floor edge in the image (perspective). Grow a window around
    # the floor until the rim reading stops rising: wide enough to reach the top edge, not so wide that it
    # takes in what stands beyond the box (on the arm it took in the wrist cable and read 388 for a 258 mm box).
    bbox, prev = None, None
    for grow in range(40, 241, 20):
        y0, y1, x0, x1 = max(0, iy.min() - grow), min(H, iy.max() + grow + 1), max(0, ix.min() - grow), min(W, ix.max() + grow + 1)
        around = wall[y0:y1, x0:x1]
        if around.sum() < 100:
            continue
        reading = robust_top(Z[y0:y1, x0:x1][around])     # the top edge; thin tall things above it do not count
        if prev is not None and reading - prev < 3.0:
            break                       # no higher than one step ago: that was the rim
        prev = rim = reading
        box_like = around & (Z[y0:y1, x0:x1] <= reading + 15.0)        # walls of the box, not what hangs above
        wx, wy = X[y0:y1, x0:x1][box_like], Y[y0:y1, x0:x1][box_like]
        bbox = (float(np.percentile(wx, 1)), float(np.percentile(wx, 99)),
                float(np.percentile(wy, 1)), float(np.percentile(wy, 99)))
    # Keep WALL_MARGIN_MM away from walls and from the edge of what was seen (quarter-resolution erosion).
    small = interior[:H // 4 * 4, :W // 4 * 4].reshape(H // 4, 4, W // 4, 4).mean(axis=(1, 3)) >= 0.7   # tolerate depth holes
    px_per_mm = K.fx / float(np.median(depth[interior]))
    k = max(1, int(round(WALL_MARGIN_MM * px_per_mm / 4)))
    while True:
        er = small
        for _ in range(k):
            er = np.logical_and.reduce(_hood(er))
        if er.any() or k <= 2:
            break
        k //= 2
    if not er.any():
        return dict(ok=False, reason="box opening too small for a safe drop", rim=rim, walls=walls)
    ey, ex = np.nonzero(er)
    px, py = X[ey * 4 + 2, ex * 4 + 2], Y[ey * 4 + 2, ex * 4 + 2]
    ok = np.isfinite(px) & np.isfinite(py)
    if not ok.any():
        return dict(ok=False, reason="no depth at the drop candidates", rim=rim, walls=walls)
    j = int(np.argmin(np.where(ok, px ** 2 + py ** 2, np.inf)))     # nearest to the arm: least reach
    return dict(ok=True, reason="", drop=np.array([px[j], py[j]]), rim=rim, walls=walls, bbox=bbox,
                margin_mm=k * 4 / px_per_mm, floor_z=float(np.median(Z[interior])))


def box_map(Z, rim: float, mark=None, K: Optional[Intrinsics] = None, X=None, Y=None) -> str:
    """80 x 30 text picture by height: '.' low, digit n = n tenths of the rim height, '#' rim, 'X' drop point."""
    H, W = Z.shape
    cw, ch = W // 80, H // 30
    cells = Z[:ch * 30, :cw * 80].reshape(30, ch, 80, cw)
    has = np.isfinite(cells).any(axis=(1, 3))
    zmax = np.nanmax(np.where(np.isfinite(cells), cells, -1e9), axis=(1, 3))
    frac = np.clip(zmax / max(rim, 1.0), 0, 1)
    rows = [[" " if not has[r, c] else "." if frac[r, c] < 0.1 else "#" if frac[r, c] >= 0.85 else str(int(frac[r, c] * 10))
             for c in range(80)] for r in range(30)]
    if mark is not None and X is not None:
        d = (X - mark[0]) ** 2 + (Y - mark[1]) ** 2
        d = np.where(np.isfinite(d), d, np.inf)
        my, mx = np.unravel_index(int(np.argmin(d)), d.shape)
        rows[min(29, my // ch)][min(79, mx // cw)] = "X"
    return "\n".join("  |" + "".join(r) + "|" for r in rows)


# ----------------------------------------------------------------------------------------------------
# Watch mode: pure decisions (no robot; tested offline on synthetic depth images)
# ----------------------------------------------------------------------------------------------------
HAND_KINDS = ("rejected:edge", "rejected:big", "rejected:tall")


def hand_blobs(objs: List[Obj], static_mask: np.ndarray, max_outside: float = 0.3) -> List[Obj]:
    """The blobs that count as a hand: big, tall or reaching in from the edge, and not there in the clean scan.
    static_mask is the raised mask of that clean scan, grown 10 px (it holds the box and anything else
    that does not move). A hand or arm shows up as an edge / big / tall blob lying mostly outside it."""
    return [o for o in objs if o.kind in HAND_KINDS and float((~static_mask[o.ys, o.xs]).mean()) > max_outside]


def hand_in_view(objs: List[Obj], static_mask: np.ndarray, max_outside: float = 0.3) -> bool:
    return bool(hand_blobs(objs, static_mask, max_outside))


def settled(prev_xy, cur_xy, tol_mm: float = 10.0, strict: bool = True, pair_mm: float = 40.0) -> bool:
    """Same parts in the same places in two frames? One-to-one match within tol_mm; needs at least one part.
    strict=False: a part with nothing within pair_mm of it in the other frame is ignored (it flickered in or
    out of one depth frame); every part that IS seen in both must be within tol_mm, and there must be one."""
    prev, cur = [np.asarray(p, float)[:2] for p in prev_xy], [np.asarray(c, float)[:2] for c in cur_xy]
    if not strict:
        pairs = 0
        for p in prev:
            d = [float(np.hypot(*(p - c))) for c in cur]
            if d and min(d) <= pair_mm:
                if min(d) > tol_mm:
                    return False
                pairs += 1
        return pairs > 0
    if not cur or len(prev) != len(cur):
        return False
    left = list(cur)
    for p in prev:
        d = [float(np.hypot(*(p - c))) for c in left]
        j = int(np.argmin(d))
        if d[j] > tol_mm:
            return False
        left.pop(j)
    return True


def expire_skips(skips: list, blob_xy: list, now: float, radius_mm: float = 40.0, ttl_s: float = 60.0) -> list:
    """Skip-list entries are (xy, time added). Keep one only while something is still within radius_mm of it
    in the current frame and it is younger than ttl_s; otherwise the spot is free to be tried again.
    An entry added at time math.inf has no age limit: it goes only when its spot is empty (parts set down
    at the reject / hold spots, which must never be picked up again)."""
    kept = []
    for xy, added in skips:
        near = any(math.hypot(xy[0] - b[0], xy[1] - b[1]) <= radius_mm for b in blob_xy)
        if near and now - added < ttl_s:
            kept.append((xy, added))
    return kept


def project_static(world_pts: np.ndarray, R: np.ndarray, t0: np.ndarray, K: Intrinsics, grow_px: int = 14) -> np.ndarray:
    """The clean scan's raised points as a mask in another camera view (for the hand rule at the close-up)."""
    mask = np.zeros((K.height, K.width), bool)
    if len(world_pts):
        pc = (world_pts - t0) @ R                     # world -> camera
        ok = pc[:, 2] > 50.0
        u = np.rint(K.fx * pc[ok, 0] / pc[ok, 2] + K.cx).astype(int)
        v = np.rint(K.fy * pc[ok, 1] / pc[ok, 2] + K.cy).astype(int)
        inside = (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
        mask[v[inside], u[inside]] = True
    return dilate(mask, grow_px)


def place_grid(spot) -> np.ndarray:
    """Every place of a spot, (n, 2) world xy, in the order they are tried: PLACE_COLS along +x, then rows away from
    the parts (+y for a spot on the +y side of the arm, -y for one on the -y side)."""
    sy = 1.0 if spot[1] >= 0 else -1.0
    return np.array([[spot[0] + PLACE_PITCH_MM * c, spot[1] + sy * PLACE_PITCH_MM * r]
                     for r in range(PLACE_ROWS) for c in range(PLACE_COLS)], float)


def place_area(spot) -> tuple:
    """The table a spot takes, (xmin, xmax, ymin, ymax): its places grown by 40 mm. Nothing is picked from it."""
    g = place_grid(spot)
    return (g[:, 0].min() - 40.0, g[:, 0].max() + 40.0, g[:, 1].min() - 40.0, g[:, 1].max() + 40.0)


def free_place(spot, taken, length_mm: float, width_mm: float) -> Optional[np.ndarray]:
    """Where the next part is set down at a spot: the first place with PLACE_GAP_MM of free table around the part and
    nothing the opening jaws would push, or None if it fits nowhere. taken: world xy, (n, 2), of what the camera sees
    lying there. The part lies with its length pointing away from the arm's base (the jaws hold a long part across,
    and set_down() turns the gripper to neutral_theta), so the jaws open across it, at its middle."""
    taken = np.asarray(taken, float).reshape(-1, 2)
    for xy in place_grid(spot):
        out = xy / max(float(np.linalg.norm(xy)), 1e-9)
        d = taken - xy
        along, across = d @ out, np.abs(d @ np.array([-out[1], out[0]]))
        body = np.hypot(along - np.clip(along, -length_mm / 2.0, length_mm / 2.0), across) >= width_mm / 2.0 + PLACE_GAP_MM
        jaws = (np.abs(along) > JAW_PAD_MM) | (across >= JAW_SWEEP_MM)
        if (body & jaws).all():
            return xy
    return None


def table_xy(info: dict, K: Intrinsics, xs, ys) -> np.ndarray:
    """World xy where the lines of sight through pixels (xs, ys) meet the table (needs info["R"], info["t"])."""
    xs, ys, z = np.asarray(xs, float), np.asarray(ys, float), info["cam_height_mm"]
    rays = np.stack([(xs - K.cx) / K.fx * z, (ys - K.cy) / K.fy * z, np.full(xs.shape, z)], axis=1)
    return (rays @ info["R"].T + info["t"])[:, :2]


def summit(info: dict, K: Intrinsics, xy, radius_mm: float = SUMMIT_RADIUS_MM) -> Optional[Tuple[np.ndarray, float, Optional[float]]]:
    """Top of the tallest thing within radius_mm of world xy -> (world xy of the top, its height above the table,
    direction in degrees the top is elongated along, or None if it is round), or None if nothing rises there. The top's own 3D points give its xy: no table-plane parallax, which puts the
    top of an 80 mm bolt 150 mm off the camera axis about 30 mm too far out."""
    h = np.nan_to_num(info["height"], nan=0.0)
    ys, xs = np.nonzero(info["raised"] & info["valid"])
    if not len(ys):
        return None
    z = info["depth"][ys, xs]
    W = np.stack([(xs - K.cx) / K.fx * z, (ys - K.cy) / K.fy * z, z], axis=1) @ info["R"].T + info["t"]
    near = np.hypot(W[:, 0] - xy[0], W[:, 1] - xy[1]) <= radius_mm
    if near.sum() < SUMMIT_MIN_PX:
        return None
    W, hh = W[near], h[ys, xs][near]
    top = float(np.sort(hh)[-SUMMIT_MIN_PX])        # not the single highest pixel: a few stray ones do not set it
    face = W[hh >= top - SUMMIT_BAND_MM, :2]
    # A tape flag on a bolt head sits as high as the head and drags the mean off the shank: report its direction.
    vals, vecs = np.linalg.eigh(np.cov(face.T)) if len(face) > 2 else (np.ones(2), np.eye(2))
    axis = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1])) if vals[1] > 2.25 * max(vals[0], 1e-9) else None
    return face.mean(axis=0), top, axis


def raised_world(info: dict, K: Intrinsics, limit: int = 20000) -> np.ndarray:
    """World points of everything a look saw standing above the table (needs info["R"], info["t"])."""
    ys, xs = np.nonzero(info["raised"] & info["valid"])
    if not len(ys):
        return np.zeros((0, 3))
    step = max(1, len(ys) // limit)
    ys, xs = ys[::step], xs[::step]
    z = info["depth"][ys, xs]
    return np.stack([(xs - K.cx) / K.fx * z, (ys - K.cy) / K.fy * z, z], axis=1) @ info["R"].T + info["t"]


def box_silhouette(bbox, rim: float, R: np.ndarray, t0: np.ndarray, K: Intrinsics, grow_mm: float = 40.0) -> np.ndarray:
    """Pixels whose line of sight passes through the box (its wall bounding box, grown, from the table to the rim)."""
    uu, vv = np.meshgrid(np.arange(K.width), np.arange(K.height))
    d = np.dstack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones(uu.shape)]) @ R.T      # ray directions, world
    lo = np.array([bbox[0] - grow_mm, bbox[2] - grow_mm, -10.0])
    hi = np.array([bbox[1] + grow_mm, bbox[3] + grow_mm, rim + 20.0])
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    t1, t2 = (lo - t0) / d, (hi - t0) / d
    tmin, tmax = np.minimum(t1, t2).max(axis=2), np.maximum(t1, t2).min(axis=2)
    return tmax >= np.maximum(tmin, 0.0)


def view_on_table(R: np.ndarray, t0: np.ndarray, K: Intrinsics, inset_mm: float = 15.0):
    """World rectangle (x0, x1, y0, y1) of the table that the camera sees whole, a little inside the image edge."""
    pts = []
    for u, v in ((0, 0), (K.width - 1, 0), (K.width - 1, K.height - 1), (0, K.height - 1)):
        ray = R @ np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
        pts.append(t0 + ray * ((TABLE_Z_EXPECTED - t0[2]) / ray[2]))
    xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
    return xs[1] + inset_mm, xs[2] - inset_mm, ys[1] + inset_mm, ys[2] - inset_mm


class MoveFailed(Exception):
    """The motion service refused or failed one move. Nothing is left moving; the run can carry on."""
    def __init__(self, what: str, reason: str):
        super().__init__(f"{what}: {reason}")
        self.what, self.reason = what, reason


def dist_to_rect(xy, rect) -> float:
    """Distance from a point to an axis-aligned rectangle (xmin, xmax, ymin, ymax); 0 inside it."""
    dx = max(rect[0] - xy[0], 0.0, xy[0] - rect[1])
    dy = max(rect[2] - xy[1], 0.0, xy[1] - rect[3])
    return math.hypot(dx, dy)


def footprint(part_xy, theta: float, t_gc) -> list:
    """The three places the hardware goes for one part, as (centre xy, clearance mm, obstacle height that matters):
    gripper body at the grasp, wrist camera at the grasp (it rides ~150 mm up), and the gripper body at the
    close-up pose (camera over the part, fingertips 186 mm up)."""
    def cam_offset(th):      # camera position relative to the gripper axis, gripper straight down at theta th
        a = math.radians(th)
        return np.array([-math.cos(a) * t_gc[0] + math.sin(a) * t_gc[1], math.sin(a) * t_gc[0] + math.cos(a) * t_gc[1]])
    part = np.asarray(part_xy[:2], float)
    return [(part, BODY_CLEAR_MM, MAX_NEIGHBOUR_MM),
            (part + cam_offset(theta), CAM_CLEAR_MM, 130.0),
            (part - cam_offset(NATURAL_THETA), BODY_CLEAR_MM, 150.0)]


def clear_of_box(bbox, part_xy, theta: float, t_gc) -> bool:
    """Would the gripper body or the wrist camera come too near the box walls while working on this part?"""
    return all(dist_to_rect(c, bbox) >= clear for c, clear, _ in footprint(part_xy, theta, t_gc))


def blocked_by_tall(info: dict, K: Intrinsics, o: Optional[Obj], part_xy, theta: float, t_gc,
                    own_radius_mm: float = 0.0) -> Optional[float]:
    """Height of something tall standing inside that footprint (a hand, a bottle, the box wall), or None.
    The part itself: its blob o, and anything within own_radius_mm of part_xy (a standing part's own top)."""
    h, depth, R, t0 = info["height"], info["depth"], info["R"], info["t"]
    own = np.zeros(h.shape, bool)
    if o is not None:
        own[o.ys, o.xs] = True
    ys, xs = np.nonzero(np.isfinite(h) & (np.nan_to_num(h, nan=0.0) >= MAX_NEIGHBOUR_MM) & ~dilate(own, 4))
    if not len(ys):
        return None
    step = max(1, len(ys) // 4000)
    ys, xs = ys[::step], xs[::step]
    z = depth[ys, xs]
    Wp = np.stack([(xs - K.cx) / K.fx * z, (ys - K.cy) / K.fy * z, z], axis=1) @ R.T + t0
    hh = h[ys, xs]
    for c, clear, hmin in footprint(part_xy, theta, t_gc):
        hit = (hh >= hmin) & (np.hypot(Wp[:, 0] - c[0], Wp[:, 1] - c[1]) < clear) \
            & (np.hypot(Wp[:, 0] - part_xy[0], Wp[:, 1] - part_xy[1]) >= own_radius_mm)
        if hit.any():
            return float(hh[hit].max())
    return None


def reachable(g) -> bool:
    """Can the gripper frame be at g with the tool pointing straight down?"""
    wc = np.array([g[0], g[1], g[2] + WRIST_ABOVE_GRIPPER - SHOULDER_Z])
    return float(np.linalg.norm(wc)) <= REACH_MM and math.hypot(g[0], g[1]) >= 150.0


def top_z(xy) -> float:
    """Highest gripper height (pointing straight down) that reaches table point xy, 10 mm inside reachable()."""
    return SHOULDER_Z - WRIST_ABOVE_GRIPPER + math.sqrt(max(REACH_MM ** 2 - xy[0] ** 2 - xy[1] ** 2, 0.0)) - 10.0


def release_tip_z(rim: float, floor_z: float, margin_mm: float) -> float:
    """World height of the fingertips when a part is let go over the box drop point: as low as the hardware
    fits, so the part is lowered in rather than dropped from carrying height. margin_mm is the room from the
    drop point to the walls: enough for the gripper body, for the open jaws only, or for neither (then the
    fingertips stop just above the rim)."""
    below = BODY_IN_BOX_MM if margin_mm >= BODY_CLEAR_MM else FINGERS_IN_BOX_MM if margin_mm >= JAW_CLEAR_MM \
        else -RIM_HOVER_MM
    return max(rim - below, max(floor_z, TABLE_Z_EXPECTED) + DROP_ABOVE_FLOOR_MM)


# ----------------------------------------------------------------------------------------------------
# Robot
# ----------------------------------------------------------------------------------------------------
class Station:
    def __init__(self, args):
        self.args = args
        self.moved = False
        self.ask = sys.stdin is not None and sys.stdin.isatty() and not args.yes
        self.gpos = None          # last known gripper position
        self.scan_g = None        # gripper position of the scan pose
        self.box = None           # dict(drop=[x, y], rim=z) once found
        self.jaws_open = False    # known open? saves an open() per part
        self.current = None       # the part being worked on, for recovery after a failed move
        self.vision = None        # anomaly vision service, if configured
        self.inspection_handler = None  # integrated pipeline: classify this exact target before grasp
        self.placed = {"box": 0, "reject": 0, "hold": 0}
        self.theta = NATURAL_THETA   # gripper theta of the last finished move
        self.released_at = None      # when the gripper last let go of a part
        self.t0 = time.time()

    async def connect(self):
        from viam.components.arm import Arm
        from viam.components.camera import Camera
        from viam.components.gripper import Gripper
        from viam.robot.client import RobotClient
        from viam.services.motion import MotionClient
        # The SDK only builds clients for resource types imported before it connects. Without these two
        # lines the capture sensor and the teleop service are "not found" later.
        import viam.components.sensor   # noqa: F401
        import viam.services.generic    # noqa: F401
        from viam.services.vision import VisionClient

        key_id, key = os.environ.get("VIAM_API_KEY_ID"), os.environ.get("VIAM_API_KEY")
        if not (key_id and key):
            sys.exit("set VIAM_API_KEY_ID and VIAM_API_KEY (machine page -> CONNECT -> API keys)")
        opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
        self.robot = await RobotClient.at_address(self.args.address, opts)
        self.arm = Arm.from_robot(self.robot, ARM)
        self.gripper = Gripper.from_robot(self.robot, GRIPPER)
        self.cam = Camera.from_robot(self.robot, CAM)
        self.motion = MotionClient.from_robot(self.robot, MOTION)
        if self.args.anomaly_service:
            self.vision = VisionClient.from_robot(self.robot, self.args.anomaly_service)
        ip = (await self.cam.get_properties()).intrinsic_parameters
        self.K = Intrinsics(ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px,
                            int(ip.width_px), int(ip.height_px))
        # Camera origin in the gripper frame; lets us place the camera by moving the gripper.
        self.t_gc = await self.transform(np.zeros(3), CAM, GRIPPER)

    async def close(self):
        await self.robot.close()

    async def transform(self, p, src: str, dst: str) -> np.ndarray:
        from viam.proto.common import Pose, PoseInFrame
        q = PoseInFrame(reference_frame=src, pose=Pose(x=float(p[0]), y=float(p[1]), z=float(p[2]), o_z=1))
        r = (await self.robot.transform_pose(q, dst)).pose
        return np.array([r.x, r.y, r.z])

    async def cam_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """Camera pose in the world as (R, t), from the machine's own frame system."""
        o = await self.transform(np.zeros(3), CAM, "world")
        axes = [(await self.transform(e, CAM, "world") - o) / 100.0 for e in np.eye(3) * 100.0]
        return np.stack(axes, axis=1), o

    async def teleop_running(self) -> bool:
        from viam.services.generic import Generic as GenericService
        try:
            status = await GenericService.from_robot(self.robot, TELEOP).do_command({"command": "status"})
            return bool(status.get("running"))
        except Exception as e:
            print(f"  WARNING: could not read teleop status ({e}). Make sure teleop is stopped yourself.")
            return False

    async def record(self, on: bool):
        """Start or stop station data capture for this run (DATA tab; one sequence per run)."""
        if self.args.record_hz <= 0:
            return
        from viam.components.sensor import Sensor
        try:
            sensor = Sensor.from_robot(self.robot, CAPTURE)
            if on:
                self.run_tag = "run-" + time.strftime("%H%M%S", time.gmtime())
                await sensor.do_command({"start_capture": True, "frequency_hz": float(self.args.record_hz),
                                         "tags": ["pick_bolt", self.run_tag]})
                print(f"  recording to the DATA tab at {self.args.record_hz:g} Hz, tags: pick_bolt, {self.run_tag}")
            else:
                await sensor.do_command({"stop_capture": True})
        except Exception as e:
            if on:
                print(f"  (not recording: {e})")

    def elapsed(self) -> float:
        return time.time() - self.t0

    async def confirm(self, what: str, gate: bool = True):
        """Announce the next action. Gated actions wait for Enter (terminal) or PAUSE_S (no terminal)."""
        print(f"\n[{self.elapsed():5.1f} s] >> NEXT: {what}", flush=True)
        if self.args.yes or not gate:
            return
        if self.ask:
            answer = await asyncio.to_thread(input, "   Enter = go, anything else = abort: ")
            if answer.strip():
                raise SystemExit("aborted by user")
        elif self.args.pause_s > 0:
            print(f"   starting in {self.args.pause_s:.0f} s (machine Stop button or E-stop to cancel)", flush=True)
            await asyncio.sleep(self.args.pause_s)

    async def capture(self, frames: Optional[int] = None, color: bool = True):
        """Several point clouds fused into one depth image, plus optional color."""
        bgr = None
        if cv2 is not None and color:
            images, _ = await self.cam.get_images()
            from robot.viam_hw import decode_color, select_image
            rgb = decode_color(select_image(images))
            bgr = np.asarray(rgb.convert('RGB'))[:, :, ::-1].copy()
        self.capture_monotonic = time.monotonic()
        depths, gray_img, in_frame, data = [], None, 1.0, b""
        for _ in range(max(1, frames or self.args.frames)):
            data, _ = await self.cam.get_point_cloud()
            pts, cloud_gray = parse_pcd(data, want_rgb=True)
            if cloud_gray is not None:
                dimg, gray_img = depth_from_points(pts, self.K, cloud_gray)
            else:
                dimg = depth_from_points(pts, self.K)
            in_frame = float((dimg > 0).sum() / max(len(pts), 1))
            depths.append(dimg)
        return fuse_depth(depths), gray_img, bgr, depths, pcd_summary(data), in_frame

    async def open_jaws(self):
        if not self.jaws_open:
            await self.confirm("open gripper", gate=False)
            await self.gripper.open()
            self.jaws_open = True

    async def look(self, tag: str, closeups: bool = True) -> Tuple[List[Obj], dict]:
        depth, gray_img, bgr, depths, summary, in_frame = await self.capture(CLOSE_FRAMES if tag == "refine" else None)
        objs, info = detect_depth(depth, self.K, bgr, gray_img)
        info["in_frame"], info["seen_at"] = in_frame, time.time()
        info['bgr'] = bgr
        info['capture_monotonic'] = self.capture_monotonic
        info["R"], info["t"] = await self.cam_frame()      # camera pose once per look; everything else is math
        cam_w, foot_w = info["t"], info["R"] @ info["foot"] + info["t"]
        info["table_z"] = float(foot_w[2])
        print(f"\n[{tag}] {len(objs)} blobs   ({len(depths)} clouds fused; each: {summary})")
        print(f"  camera at world x={cam_w[0]:.0f} y={cam_w[1]:.0f} z={cam_w[2]:.0f}; table plane under it at world "
              f"z={foot_w[2]:.1f} (expected about {TABLE_Z_EXPECTED:.0f})")
        text = report(objs, info, self.K)
        print(text if closeups else text.split("\n  view (")[0], flush=True)     # sort mode: table only
        if closeups:   # what the depth really looks like around the most interesting blobs
            keep = [o for o in objs if not o.kind.startswith("rejected") and o.height_mm is not None]
            for o in sorted(keep, key=lambda o: (-o.score, -o.height_mm))[:2]:
                print(f"  blob {o.idx} ({o.kind}):")
                print(zoom_map(info, o.u, o.v), flush=True)
        if in_frame < 0.6:
            print("  WARNING: point cloud does not line up with the color camera; heights are unreliable")
        try:   # extras for a terminal run; the app sandbox may have nowhere to write
            base = os.path.join(self.args.debug_dir, f"pick_bolt_{tag}")
            if bgr is not None:
                cv2.imwrite(base + ".jpg", annotate(bgr, objs))
                print(f"  annotated image: {base}.jpg")
            if self.args.save_raw:
                np.savez_compressed(base + ".npz", depths=np.stack(depths), bgr=bgr if bgr is not None else 0,
                                    K=np.array([self.K.fx, self.K.fy, self.K.cx, self.K.cy]))
        except OSError as e:
            print(f"  (could not save debug files: {e})")
        return objs, info

    async def glance(self, pose) -> Tuple[List[Obj], dict]:
        """One quiet single-frame look for watch mode: no report, no files. pose = the camera's (R, t), which
        the caller knows because the arm has not moved since it last asked."""
        depth, gray_img, *_ = await self.capture(1, color=False)
        seen = time.time()
        objs, info = detect_depth(depth, self.K, None, gray_img)
        info["R"], info["t"], info["seen_at"] = pose[0], pose[1], seen
        return objs, info

    async def look_world(self):
        """World-frame X, Y, Z images of what the camera sees now."""
        depth, *_ = await self.capture()
        R, t = await self.cam_frame()
        X, Y, Z = world_maps(depth, self.K, R, t)
        return X, Y, Z, depth, t

    async def go_home(self, gate: bool = True):
        from viam.proto.component.arm import JointPositions
        await self.confirm(f"joint move to home {HOME_JOINTS_DEG} (tool straight down)", gate)
        self.moved = True
        await self.arm.move_to_joint_positions(JointPositions(values=HOME_JOINTS_DEG), timeout=60)
        self.theta = NATURAL_THETA
        self.gpos = await self.transform(np.zeros(3), GRIPPER, "world")
        cam_w = await self.transform(np.zeros(3), CAM, "world")
        self.scan_g = self.gripper_xyz_for_camera([cam_w[0], cam_w[1], TABLE_Z_EXPECTED + SCAN_HEIGHT_MM])

    async def move_gripper(self, xyz, theta: float, what: str, linear: bool = False, gate: bool = False):
        from viam.proto.common import Pose, PoseInFrame
        from viam.proto.service.motion import Constraints, LinearConstraint
        x, y, z = (float(c) for c in xyz)
        floor = TABLE_Z_EXPECTED + self.args.tip_offset_mm - 3      # fingertips never below the expected table
        if not (60 <= x <= 650 and -420 <= y <= 420 and z >= floor):
            raise MoveFailed(what, f"gripper goal x={x:.0f} y={y:.0f} z={z:.0f} is outside sane limits (z floor {floor:.0f})")
        await self.confirm(f"{what}: gripper -> x={x:.0f} y={y:.0f} z={z:.0f} theta={theta:.0f}"
                           + (" in a straight line" if linear else ""), gate)
        constraints = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=3,
                                                                      orientation_tolerance_degs=3)]) if linear else None
        dest = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=theta))
        self.moved = True
        try:
            ok = await self.motion.move(component_name=GRIPPER, destination=dest, constraints=constraints, timeout=120)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:   # planner refusal, unreachable goal, ...: the arm has not moved
            raise MoveFailed(what, str(getattr(e, "message", None) or e)) from e
        if not ok:
            raise MoveFailed(what, "motion service reported failure")
        self.gpos, self.theta = np.array([x, y, z]), theta

    async def travel(self, xy, z: float, theta: float, what: str, gate: bool = False):
        """Straight-line trip that does its sideways part at the higher of the two heights."""
        if self.gpos is None:
            self.gpos = await self.transform(np.zeros(3), GRIPPER, "world")
        x0, y0, z0 = self.gpos
        same_xy = math.hypot(xy[0] - x0, xy[1] - y0) < 2.0
        if z > z0 + 2.0:
            await self.move_gripper([x0, y0, z], theta, f"{what} (up)", linear=True, gate=gate)
            gate = False
        if not same_xy:
            await self.move_gripper([xy[0], xy[1], self.gpos[2]], theta, f"{what} (across)", linear=True, gate=gate)
        if abs(self.gpos[2] - z) > 2.0:
            await self.move_gripper([xy[0], xy[1], z], theta, f"{what} (down)", linear=True)

    async def go_scan(self, gate: bool = False):
        await self.travel(self.scan_g[:2], self.scan_g[2], NATURAL_THETA, f"scan pose, camera {SCAN_HEIGHT_MM:.0f} mm up", gate)

    def gripper_xyz_for_camera(self, cam_xyz) -> np.ndarray:
        """Gripper position (straight down, natural theta) that puts the camera at cam_xyz."""
        t = math.radians(NATURAL_THETA)
        R = np.array([[-math.cos(t), math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, -1.0]])
        return np.asarray(cam_xyz, float) - R @ self.t_gc

    def locate(self, o: Obj, info: dict) -> Tuple[np.ndarray, float]:
        """World position of the table point under the object, and its long-axis direction in degrees."""
        base, tip = info["R"] @ o.p_table + info["t"], info["R"] @ o.p_axis + info["t"]
        return base, math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0]))


async def find_box(st: Station) -> bool:
    """From home: spot the box, look into it from above, choose the drop point. Sets st.box."""
    rim_guess, look_xy = 250.0, None
    if st.args.box_hint:
        look_xy = np.array(st.args.box_hint, float)
        print(f"\n[box] using BOX_HINT_XY {look_xy.tolist()}")
    else:
        X, Y, Z, depth, cam_t = await st.look_world()
        far = find_tall_object(X, Y, Z, depth, st.K, cam_t[:2])
        if far is None:
            print(f"\n[box] nothing at least {BOX_MIN_HEIGHT_MM:.0f} mm tall and box-sized in view from home.")
            print(box_map(Z, 200.0))
            print("  Move the box into view (it must show within about 200 mm of the home camera spot), or set "
                  "BOX_HINT_XY = (x, y) of a point inside it.")
            return False
        rim_guess, look_xy = far["rim"], far["look_xy"]
        print(f"\n[box] tall object from home: about {far['rim']:.0f} mm high, nearest rim at x={far['near'][0]:.0f} "
              f"y={far['near'][1]:.0f}; looking into it at x={look_xy[0]:.0f} y={look_xy[1]:.0f}")
        print(box_map(Z, far["rim"], look_xy, st.K, X, Y), flush=True)
    # Highest reachable camera spot over that point; fingertips stay well above the rim on the way.
    g = None
    for cam_z in np.arange(BOX_LOOK_CAM_Z, max(rim_guess + 219.0, 499.0), -20.0):   # fingertips at cam_z - 154
        cand = st.gripper_xyz_for_camera([look_xy[0], look_xy[1], cam_z])
        if reachable(cand):
            g = cand
            break
    if g is None:
        print("  The box is too far away to look into from above. Move it closer to the arm.")
        return False
    # Down first (at the current spot), then straight across: never sideways below this height.
    await st.move_gripper([st.gpos[0], st.gpos[1], g[2]], NATURAL_THETA, "height for the box look", linear=True, gate=True)
    await st.move_gripper(g, NATURAL_THETA, "camera over the box", linear=True)
    X, Y, Z, depth, cam_t = await st.look_world()
    res = find_drop_point(X, Y, Z, depth, st.K, rim_guess)
    rim = res.get("rim", rim_guess)
    if not st.args.box_hint:
        rim = max(rim, rim_guess)          # two measurements of the rim: believe the higher one
    print(f"\n[box] from above (camera z={cam_t[2]:.0f}):")
    print(box_map(Z, rim, res.get("drop"), st.K, X, Y), flush=True)
    if not res["ok"]:
        print(f"  Not accepted as a box: {res['reason']}. Nothing will be dropped.")
        return False
    drop = res["drop"]
    drop_z = None
    for clear in (DROP_CLEAR_MM, 70.0, 50.0):
        z = rim + clear + st.args.tip_offset_mm
        if reachable([drop[0], drop[1], z]):
            drop_z = z
            break
    if drop_z is None:
        print(f"  Box found (rim {rim:.0f} mm) but the drop point x={drop[0]:.0f} y={drop[1]:.0f} is out of reach "
              "at a safe height. Move the box closer to the arm.")
        return False
    bbox = res.get("bbox") or (drop[0] - 100, drop[0] + 100, drop[1] - 100, drop[1] + 100)
    # Where to let go: lowered into the box as far as the hardware fits, never above the carrying height.
    release_z = min(drop_z, release_tip_z(rim, res["floor_z"], res["margin_mm"]) + st.args.tip_offset_mm)
    while release_z < drop_z and not reachable([drop[0], drop[1], release_z]):
        release_z = min(drop_z, release_z + 20.0)
    st.box = dict(drop=drop, rim=rim, drop_z=drop_z, release_z=release_z, bbox=bbox)
    tip = release_z - st.args.tip_offset_mm
    print(f"  BOX OK: rim {rim:.0f} mm high, floor at z={res['floor_z']:.0f}, walls seen: {', '.join(res['walls'])}. "
          f"Drop point x={drop[0]:.0f} y={drop[1]:.0f} ('X' above), {res['margin_mm']:.0f} mm from the walls; "
          f"gripper carries at z={drop_z:.0f} (fingertips {drop_z - st.args.tip_offset_mm - rim:.0f} mm over the rim) "
          f"and lets go with the fingertips {abs(tip - rim):.0f} mm {'below' if tip < rim else 'above'} the rim, "
          f"{tip - res['floor_z']:.0f} mm above the box floor.")
    return True


async def pick_one(st: Station, stop_at: str, skip: list, force_close_up: bool = False, guard=None,
                   target_world_xy=None, target_tolerance_mm=None, roi_target=None, target_height_mm=None) -> dict:
    """One part: look, choose, (close-up,) grasp, lift. stop_at is 'look', 'hover' or 'pick'.
    Returns dict(status=none|out|look|lost|refused|hover|held|missed|hand, ...).
    Watch mode passes force_close_up and guard(objs, info, tag): called on the scan look before anything
    moves and on the close-up look before the gripper goes down; a reason string from it stops the pick."""
    args = st.args
    objs, info = await st.look("scan", closeups=not st.box)
    if guard is not None:
        why = guard(objs, info, "scan")
        if why:
            print(f"\nHand rule: {why}. Not moving.")
            return dict(status="hand", why=why)
    if target_world_xy is not None and "height" in info:   # a standing part is never matched by its ROI: aim at its top
        up = summit(info, st.K, target_world_xy)
        if up is not None:
            print(f"  standing check: top {up[1]:.0f} mm up, {np.linalg.norm(up[0] - np.asarray(target_world_xy)[:2]):.0f} mm "
                  f"from the target (standing if >= {STANDING_MIN_TOP_MM:.0f} mm up and <= {STANDING_AIM_MM:.0f} mm off)")
        if up is not None and up[1] >= STANDING_MIN_TOP_MM \
                and np.linalg.norm(up[0] - np.asarray(target_world_xy)[:2]) <= STANDING_AIM_MM:
            return await pick_standing(st, stop_at, info, objs, up[0], up[1], up[2])
        # A thin standing screw's top can give the scan too little depth to find (14 Nov: 72 mm screw, 'insufficient
        # depth', then the nut next to it refused as 'too close to something tall'). The home capture measured it in
        # 3D already: aim there.
        if target_height_mm is not None and target_height_mm >= STANDING_MIN_TOP_MM:
            print(f"  standing (from the home capture): {target_height_mm:.0f} mm tall at the target position")
            return await pick_standing(st, stop_at, info, objs, np.asarray(target_world_xy, float)[:2], float(target_height_mm))
    cands = [o for o in objs if o.kind in PICK_KINDS]
    if roi_target is not None:
        try:cands=[roi_target(info)]
        except ValueError as exc:
            return dict(status='target_unmatched',xy=np.asarray(target_world_xy),why=str(exc))
        from robot.roi_pick_target import validate_approach_target
        refusal=validate_approach_target(cands[0])
        if refusal:return dict(status='refused',xy=np.asarray(target_world_xy),why=refusal)
    if args.index is not None:
        cands = [o for o in objs if o.idx == args.index]
    if target_world_xy is not None:
        if target_tolerance_mm is None or not math.isfinite(target_tolerance_mm) or target_tolerance_mm<=0:
            raise ValueError('External target requires a positive matching tolerance')
        cands=[o for o in cands if np.linalg.norm(st.locate(o,info)[0][:2]-target_world_xy)<=target_tolerance_mm]
        if len(cands)!=1:
            return dict(status='target_unmatched',xy=np.asarray(target_world_xy),
                        why=f'External target has {len(cands)} scan matches; no arbitrary substitution')
    target = base = None
    for o in sorted(cands, key=lambda o: -o.area_mm2):     # biggest first: easiest to see and to hold
        b, axis_deg = st.locate(o, info)
        if any(math.hypot(b[0] - s[0], b[1] - s[1]) < 40 for s in skip):
            continue
        target, base = o, b
        break
    if target is None:
        print("\nNothing pickable left in view." + ("" if skip else " If a part is listed above under another "
              "label, set PICK_INDEX (or --index) to its number."))
        left = [o for o in objs if not o.kind.startswith("rejected") and o.kind not in PICK_KINDS]
        return dict(status="none", left=[(o.kind, o.length_mm, o.width_mm, o.height_mm or 0.0, o.u, o.v) for o in left[:8]])
    others = [o.idx for o in cands if o.idx != target.idx]
    print(f"\nTarget {target.idx} ({target.kind}): world x={base[0]:.0f} y={base[1]:.0f}, table z={base[2]:.0f}"
          + (f"   (also pickable: {others})" if others else ""))
    out = dict(xy=base[:2].copy(), scan_xy=base[:2].copy(), kind=target.kind, score=target.score, seen_at=info["seen_at"],
               size=(target.length_mm, target.width_mm, target.height_mm or 0.0),
               decision="box", anomaly=None, confidence=target.score)
    for axis, val in (("x", base[0]), ("y", base[1])):
        lo, hi = WORKSPACE[axis]
        if not lo <= val <= hi:
            print(f"Target {axis}={val:.0f} is outside the workspace {lo:.0f}..{hi:.0f}. Not moving.")
            return dict(out, status="out")
    if not (TABLE_Z_RANGE[0] <= base[2] <= TABLE_Z_RANGE[1]) and not args.force:
        print(f"Table measured at z={base[2]:.0f}, outside {TABLE_Z_RANGE[0]:.0f}..{TABLE_Z_RANGE[1]:.0f}. "
              "Camera calibration or depth is off. Not moving.")
        return dict(out, status="out")
    if stop_at == "look":
        print('Look only. Next step: MODE = "hover".')
        return dict(out, status="look")
    st.current = out
    if target_world_xy is None and target.kind == "bolt-standing":
        up = summit(info, st.K, base)
        if up is not None and up[1] >= STANDING_MIN_TOP_MM:
            return await pick_standing(st, stop_at, info, objs, up[0], up[1], up[2])
    # Check the surroundings before going anywhere near: the close-up pose already has the fingertips at 186 mm,
    # lower than the box rim.
    scan_plan = grasp_plan(target.length_mm, target.width_mm, target.height_mm, tip_clearance_mm=args.tip_clearance_mm)
    if scan_plan is None:      # only a blob chosen by hand gets here: pickable kinds all have a plan
        print(f"{target.length_mm:.0f} x {target.width_mm:.0f} mm does not fit the open jaws "
              f"(GRIP_MAX_SPAN_MM = {GRIP_MAX_SPAN_MM:.0f}). Not approaching.")
        return dict(out, status="refused")
    plan_theta = grasp_theta(axis_deg, args.jaw_axis, near=neutral_theta(base[0], base[1])) if scan_plan[0] \
        else neutral_theta(base[0], base[1])
    tall = blocked_by_tall(info, st.K, target, base, plan_theta, st.t_gc)
    near_box = st.box is not None and not clear_of_box(st.box["bbox"], base, plan_theta, st.t_gc)
    if tall is not None or near_box:
        why = "the box for the gripper and camera to fit" if near_box else f"something {tall:.0f} mm tall"
        print(f"Too close to {why}. Not approaching.")
        return dict(out, status="refused")

    scan_kind, scan_h = target.kind, target.height_mm
    out["first_motion_at"] = time.time()
    # A long lying bolt is measured well enough from the scan height; going straight in saves a move and a look.
    # (Not with an anomaly model: that needs the close-up view of every part.)
    direct = (roi_target is None and not force_close_up and args.close_up == "auto" and stop_at == "pick" and args.index is None and st.vision is None
              and target.kind == "bolt-lying" and target.length_mm >= 40 and target.height_mm is not None)
    if not direct:
        # ---- close-up from straight above ----
        cam_goal = [base[0], base[1], TABLE_Z_EXPECTED + REFINE_HEIGHT_MM]
        await st.move_gripper(st.gripper_xyz_for_camera(cam_goal), NATURAL_THETA, "camera above the part", gate=not st.box)
        objs, info = await st.look("refine", closeups=not st.box)
        if roi_target is not None:
            try:objs=[roi_target(info)]
            except ValueError as exc:return dict(out,status='target_unmatched',why=str(exc))
            from robot.roi_pick_target import validate_grasp_target
            refusal=validate_grasp_target(objs[0])
            if refusal:
                print(refusal+'. Not grasping.')
                return dict(out,status='refused',why=refusal)
        if guard is not None:
            why = guard(objs, info, "refine")
            if why:
                print(f"\nHand rule at the close-up: {why}. Not descending.")
                return dict(out, status="hand", why=why)
        off_center = lambda o: math.hypot(o.u - st.K.cx, o.v - st.K.cy)  # noqa: E731
        near = [o for o in objs if not o.kind.startswith("rejected") and off_center(o) < 90]
        if target_world_xy is not None:
            near=[o for o in near if np.linalg.norm(st.locate(o,info)[0][:2]-target_world_xy)<=target_tolerance_mm]
            if len([o for o in near if o.kind in PICK_KINDS])>1:
                return dict(out,status='target_unmatched',why='Ambiguous external target at close-up')
        if not near:
            print("Lost the part in the close-up. Not grasping.")
            return dict(out, status="lost")
        # Prefer something pickable, unless the blob was chosen by hand.
        target = min(near, key=lambda o: (0 if args.index is not None or o.kind in PICK_KINDS else 1, off_center(o)))
        if args.index is None and target.kind not in PICK_KINDS:
            # Straight above a shiny low part the depth can drop out entirely. If the scan measured a nut
            # here and the close-up still shows a nut-sized dark patch, that is two independent sightings.
            alt = [o for o in near if o.kind == "dark" and 8 <= o.length_mm <= NUT_MAX_LEN and off_center(o) < 60]
            if scan_kind == "nut" and "nut" in PICK_KINDS and alt:
                target = min(alt, key=off_center)
                # Weak close-up depth means it is thin, whatever the scan read: use the lowest fingertip height.
                target.kind, target.height_mm = "nut", min(scan_h or THIN_PART_MM, THIN_PART_MM - 0.5)
                print("  (close-up depth too weak on this part: position from its color patch, height from the scan)")
            else:
                print(f"Close-up does not confirm a pickable part there (nearest blob: {target.kind}, "
                      f"{target.length_mm:.0f} x {target.width_mm:.0f} x {target.height_mm or 0:.0f} mm). Not grasping.")
                return dict(out, status="lost")
        base, axis_deg = st.locate(target, info)
    # Jaw direction and fingertip height come from the measured size, not from what the part was called.
    plan = grasp_plan(target.length_mm, target.width_mm, target.height_mm, upright=scan_plan[1] if roi_target is None else False,
                      tip_clearance_mm=args.tip_clearance_mm)
    if plan is None:
        print(f"Close-up measures {target.length_mm:.0f} x {target.width_mm:.0f} mm: does not fit the open jaws "
              f"(GRIP_MAX_SPAN_MM = {GRIP_MAX_SPAN_MM:.0f}). Not grasping.")
        return dict(out, status="refused")
    lying, standing, tip_above_table = plan
    # Of the two jaw directions, take the one that keeps the camera on the outward side of the wrist.
    outward = neutral_theta(base[0], base[1])
    theta = grasp_theta(axis_deg, args.jaw_axis, near=outward) if lying else outward
    tall = blocked_by_tall(info, st.K, target, base, theta, st.t_gc)
    if tall is not None:
        print(f"Something {tall:.0f} mm tall is where the gripper or camera would go (hand? bottle? box wall?). Not grasping.")
        return dict(out, status="refused")
    obj_h = target.height_mm or 0.0
    if standing:
        obj_h = max(obj_h, scan_h or 0.0)          # for the log only: grasp_plan does not use an upright part's height
    # The arm stands on the table, so the table is at z = 0; the wrist depth reads it a few mm high (more
    # from further away). Take that known over-read off, then keep within 2 mm of where the table must be.
    table_z = base[2] - TABLE_READS_HIGH["scan" if direct else "close"]
    table_z = min(max(table_z, TABLE_Z_EXPECTED - 2.0), TABLE_Z_EXPECTED + 2.0)
    grasp_z = table_z + tip_above_table + args.tip_offset_mm
    out.update(xy=base[:2].copy(), kind=target.kind, score=max(out["score"], target.score), theta=theta, grasp_z=grasp_z,
               size=(target.length_mm, target.width_mm, obj_h))
    print(f"\n{'From the scan' if direct else 'Close-up'}: {target.kind}, {target.length_mm:.0f} x {target.width_mm:.0f} x "
          f"{obj_h:.0f} mm at x={base[0]:.1f} y={base[1]:.1f}, table z={base[2]:.1f} (using {table_z:.1f}), "
          f"long axis {axis_deg:.0f} deg")
    print(f"Grasp: theta={theta:.0f}, fingertips {tip_above_table:.0f} mm above the table "
          f"(gripper frame z={grasp_z:.0f}, tip offset {args.tip_offset_mm:.0f})")

    # ---- decide where it goes, while the camera can still see it (it cannot see a part in the jaws) ----
    out["decision"], out["anomaly"], out["confidence"] = await decide(st, out, target, info, objs)
    if st.inspection_handler is not None and out['decision']=='hold':
        print('Inspection HOLD: leave target unpicked; no grasp or delivery.')
        return dict(out,status='inspection_hold')
    if out["anomaly"] is not None:
        print(f"Decision: {out['decision']} (anomaly score {out['anomaly']:.2f})")

    # ---- grasp ----
    await st.open_jaws()
    await st.move_gripper([base[0], base[1], grasp_z + APPROACH_MM], theta, "pre-grasp above the part")
    if stop_at == "hover":
        print(f"\nHover only. Check on the arm now:\n"
              f"  - fingertips should be {tip_above_table + APPROACH_MM:.0f} mm above the table "
              f"(if not, change TIP_OFFSET_MM by the difference)\n"
              f"  - the part should sit centered between the jaws"
              + ('\n  - the jaws should close ACROSS the bolt (if they are along it, set JAW_AXIS = "x")' if lying else "")
              + '\nIf all good: MODE = "pick".')
        return dict(out, status="hover")
    await st.move_gripper([base[0], base[1], grasp_z], theta, "descend to grasp", linear=True, gate=not st.box)
    await st.confirm("close gripper", gate=False)
    grabbed = await st.gripper.grab()
    st.jaws_open = False
    lift_z = grasp_z + LIFT_MM
    if st.box:   # go straight to carrying height if this spot allows it
        lift_z = max(lift_z, st.box["drop_z"])
        while lift_z > grasp_z + LIFT_MM and not reachable([base[0], base[1], lift_z]):
            lift_z -= 20.0
    await st.move_gripper([base[0], base[1], lift_z], theta, "lift", linear=True)
    holding = await st.gripper.is_holding_something()
    print(f"\ngrab() returned {grabbed}; is_holding_something: {holding.is_holding_something} {holding.meta or ''}")
    # The gripper's own verdict decides: jaws that closed all the way hold nothing.
    return dict(out, status="held" if holding.is_holding_something else "missed")


async def pick_standing(st: Station, stop_at: str, info: dict, objs: list, top_xy, top_h: float,
                       top_axis_deg: Optional[float] = None) -> dict:
    """A standing part, from the scan: open jaws over its top, straight down around it, close, lift. No close-up
    (too near for depth on the top). Fingertips go to STANDING_TIP_MM, higher for parts taller than the gripper
    body clears there, and arrive from TOP_CLEAR_MM above the top."""
    args = st.args
    x, y = float(top_xy[0]), float(top_xy[1])
    # A round top: any jaw direction; this one keeps the camera outward. An elongated top (a tape flag): the jaws
    # close ALONG it, so the closing jaws slide the shank to the middle however far the flag pulled the aim off.
    theta = neutral_theta(x, y) if top_axis_deg is None else \
        grasp_theta(top_axis_deg + 90.0, args.jaw_axis, near=neutral_theta(x, y))
    tip = STANDING_TIP_MM + max(0.0, top_h - STANDING_BODY_CLEAR_MM)
    grasp_z = TABLE_Z_EXPECTED + tip + args.tip_offset_mm
    hover_z = TABLE_Z_EXPECTED + max(tip + APPROACH_MM, top_h + TOP_CLEAR_MM) + args.tip_offset_mm
    out = dict(xy=np.array([x, y]), scan_xy=np.array([x, y]), kind="bolt-standing", score=1.0, seen_at=info["seen_at"],
               size=(15.0, 15.0, top_h), decision="box", anomaly=None, confidence=1.0, theta=theta, grasp_z=grasp_z)
    print(f"\nStanding part: top {top_h:.0f} mm up at x={x:.1f} y={y:.1f}. Fingertips down to {tip:.0f} mm, "
          f"arriving from {hover_z - args.tip_offset_mm:.0f} mm (theta {theta:.0f})")
    for axis, val in (("x", x), ("y", y)):
        lo, hi = WORKSPACE[axis]
        if not lo <= val <= hi:
            print(f"Target {axis}={val:.0f} is outside the workspace {lo:.0f}..{hi:.0f}. Not moving.")
            return dict(out, status="out")
    tall = blocked_by_tall(info, st.K, None, (x, y), theta, st.t_gc, own_radius_mm=STANDING_OWN_MM)
    near_box = st.box is not None and not clear_of_box(st.box["bbox"], (x, y), theta, st.t_gc)
    if tall is not None or near_box:
        print(f"Too close to {'the box' if near_box else f'something {tall:.0f} mm tall'}. Not approaching.")
        return dict(out, status="refused")
    if stop_at == "look":
        return dict(out, status="look")
    st.current = out
    out["first_motion_at"] = time.time()
    out["decision"], out["anomaly"], out["confidence"] = await decide(st, out, None, info, objs)
    if st.inspection_handler is not None and out["decision"] == "hold":
        print("Inspection HOLD: leave target unpicked; no grasp or delivery.")
        return dict(out, status="inspection_hold")
    await st.open_jaws()
    await st.move_gripper([x, y, hover_z], theta, "pre-grasp above the standing part")
    if stop_at == "hover":
        return dict(out, status="hover")
    await st.move_gripper([x, y, grasp_z], theta, "descend around the standing part", linear=True, gate=not st.box)
    await st.confirm("close gripper", gate=False)
    grabbed = await st.gripper.grab()
    st.jaws_open = False
    lift_z = grasp_z + LIFT_MM
    if st.box:
        lift_z = max(lift_z, st.box["drop_z"])
        while lift_z > grasp_z + LIFT_MM and not reachable([x, y, lift_z]):
            lift_z -= 20.0
    await st.move_gripper([x, y, lift_z], theta, "lift", linear=True)
    holding = await st.gripper.is_holding_something()
    print(f"\ngrab() returned {grabbed}; is_holding_something: {holding.is_holding_something} {holding.meta or ''}")
    return dict(out, status="held" if holding.is_holding_something else "missed")


async def decide(st: Station, item: dict, target=None, info=None, objects=None) -> Tuple[str, Optional[float], float]:
    """Where does this part go? Returns (decision, anomaly score or None, confidence 0..1).
    decision is "box", "reject" or "hold". Runs while the camera is over the part, before the grasp.
    With no anomaly service configured this is a stub: everything is "box"."""
    handler=getattr(st,'inspection_handler',None)
    if handler is not None:
        result=await handler(st,item,target,info,objects)
        if not isinstance(result,tuple) or len(result)!=3 or result[0] not in ('box','reject','hold'):
            raise ValueError('Invalid integrated inspection result; refusing grasp')
        return result
    if st.vision is None:
        return "box", None, float(item["score"])
    try:
        best = (await st.vision.get_classifications_from_camera(CAM, 1))[0]
        # viam-labs:anomalib:vision: "anomalous" with confidence = score, or "normal" with confidence = 1 - score
        score = float(best.confidence if best.class_name == "anomalous" else 1.0 - best.confidence)
    except Exception as e:
        print(f"  anomaly service failed ({e}); sending this part to hold")
        return "hold", None, 0.0
    if score >= ANOMALY_REJECT:
        return "reject", score, score
    if score >= ANOMALY_HOLD:
        return "hold", score, 0.5
    return "box", score, 1.0 - score


def _closest_to_base(p0, p1) -> float:
    """Smallest distance from the arm's base axis to the straight segment p0 -> p1 (XY)."""
    p0, p1 = np.asarray(p0[:2], float), np.asarray(p1[:2], float)
    d = p1 - p0
    s = float(np.clip(-(p0 @ d) / max(d @ d, 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p0 + s * d))


async def drop_in_box(st: Station):
    """Carry what is held to the box and let go. Sideways moves happen at carrying height only, and the
    wrist turns back to neutral on the way (camera outward), whatever angle the grasp needed."""
    box = st.box
    over_box = [box["drop"][0], box["drop"][1], box["drop_z"]]
    done = False
    if st.gpos[2] >= box["drop_z"] - 2.0 and _closest_to_base(st.gpos, over_box) >= 200.0:
        try:
            await st.move_gripper(over_box, neutral_theta(*box["drop"]), "carry: straight over the box", linear=True)
            done = True
        except MoveFailed as e:
            print(f"  straight carry refused ({e.reason}); going via the scan spot")
    if not done:   # could not lift all the way out there, the line passes too near the base, or the planner said no
        await st.move_gripper([st.scan_g[0], st.scan_g[1], box["drop_z"]], NATURAL_THETA,
                              "carry: to the scan spot at carrying height", linear=True)
        await st.move_gripper(over_box, neutral_theta(*box["drop"]), "carry: over the box", linear=True)
    # Straight down into the box before letting go, so the part is set in rather than dropped from carrying height.
    release = [box["drop"][0], box["drop"][1], box.get("release_z", box["drop_z"])]
    lower = release[2] < box["drop_z"] - 2.0
    if lower:
        try:
            await st.move_gripper(release, neutral_theta(*box["drop"]), "lower into the box", linear=True)
        except MoveFailed as e:
            print(f"  lowering into the box refused ({e.reason}); letting go from where the gripper is")
    await st.confirm(f"open gripper {'in' if lower else 'over'} the box (x={box['drop'][0]:.0f} y={box['drop'][1]:.0f})", gate=False)
    await st.gripper.open()
    st.jaws_open, st.released_at = True, time.time()
    await asyncio.sleep(0.25)
    if lower:      # the same line back up: nothing sideways below the carrying height
        await st.move_gripper(over_box, neutral_theta(*box["drop"]), "straight up out of the box", linear=True)
    # Back to the scan pose in one slanted line if the fingertips still clear the rim where they cross the wall.
    tip0, tip1 = box["drop_z"] - st.args.tip_offset_mm, st.scan_g[2] - st.args.tip_offset_mm
    run = max(math.hypot(st.scan_g[0] - box["drop"][0], st.scan_g[1] - box["drop"][1]), 1.0)
    tip_at_wall = tip0 + (tip1 - tip0) * min(1.0, (WALL_MARGIN_MM + 40.0) / run)
    if tip_at_wall >= box["rim"] + 50.0:
        await st.move_gripper(st.scan_g, NATURAL_THETA, f"back to the scan pose, camera {SCAN_HEIGHT_MM:.0f} mm up", linear=True)
    else:
        await st.go_scan()


async def carry_over(st: Station, xy, theta: float, what: str) -> float:
    """Carry what is held, at carrying height, until the gripper is over table point xy. Returns that height: lower
    than the scan pose for a point too far out to reach up there (see top_z)."""
    carry_z = min(max(st.gpos[2], st.scan_g[2]), top_z(xy))
    if math.hypot(xy[0] - st.gpos[0], xy[1] - st.gpos[1]) > 2.0:
        if _closest_to_base(st.gpos, xy) < 200.0 or st.gpos[2] < carry_z - 2.0:
            await st.move_gripper([st.scan_g[0], st.scan_g[1], carry_z], NATURAL_THETA, f"{what}: to the scan spot", linear=True)
        await st.move_gripper([xy[0], xy[1], carry_z], theta, f"{what}: over the spot", linear=True)
    return carry_z


async def lying_at(st: Station, spot, what: str) -> np.ndarray:
    """Carry what is held to where the camera sees the places of a spot. Returns world xy, (n, 2), of everything
    lying there. The table is the only record of what earlier runs left: counting places instead put the first
    part of every run on top of the last run's (14 Nov: two runs in a row both started at x=300 y=250)."""
    middle = place_grid(spot).mean(axis=0)
    await carry_over(st, st.gripper_xyz_for_camera([middle[0], middle[1], 0.0]), NATURAL_THETA, f"{what}: camera")
    objs, info = await st.look("place", closeups=False)
    seen = raised_world(info, st.K)
    seen = seen[seen[:, 2] < TABLE_Z_EXPECTED + BOLT_MAX_H][:, :2]      # not the part in the jaws, nor the jaws
    # Every blob as well: a nut set down at x=300 y=250 read 4 mm high, under the raised cut, and the next part
    # went on top of it (14 Nov, 16:27 UTC). Low, flat and color-only blobs are taken, wherever they are.
    low = [table_xy(info, st.K, o.xs, o.ys) for o in objs if o.xs is not None and (o.height_mm or 0.0) < BOLT_MAX_H
           and (o.height_mm is None or o.height_mm > TAPE_MAX_MM)]
    return np.vstack([seen] + low)


async def set_down(st: Station, xy, z: float, what: str) -> bool:
    """Carry what is held (at the current height) over a table spot, lower it, let go, come back up.
    False if the part fell out of the jaws on the way: then nothing is lowered and the jaws stay shut."""
    theta = neutral_theta(xy[0], xy[1])
    carry_z = await carry_over(st, xy, theta, what)
    if not (await st.gripper.is_holding_something()).is_holding_something:
        print(f"  {what}: the part is no longer in the jaws: it fell on the way. Not lowering; the next look finds it.")
        if carry_z < st.scan_g[2] - 2.0:
            await st.move_gripper(st.scan_g, NATURAL_THETA, f"{what}: back to the scan pose", linear=True)
        return False
    await st.move_gripper([xy[0], xy[1], z], theta, f"{what}: lower", linear=True)
    await st.confirm("open gripper", gate=False)
    await st.gripper.open()
    st.jaws_open, st.released_at = True, time.time()
    await st.move_gripper([xy[0], xy[1], carry_z], theta, f"{what}: back up", linear=True)
    if carry_z < st.scan_g[2] - 2.0:     # too far out to rise in place: back to the scan pose in one slanted line
        await st.move_gripper(st.scan_g, NATURAL_THETA, f"{what}: back to the scan pose", linear=True)
    return True


async def deliver(st: Station, res: dict, skip: list) -> str:
    """Take the held part where its decision says. Returns a short result for the log."""
    dec = res["decision"]
    if dec == "box" and st.args.good_xy is None:
        await drop_in_box(st)
        return "in box"
    spot = {"box": st.args.good_xy, "reject": st.args.reject_xy, "hold": st.args.hold_xy}[dec]
    if spot is None:   # nowhere to put it: never into the good box. Back where it came from.
        print(f"  decision is {dec} but {dec.upper()}_XY is not set: putting the part back")
        await set_down(st, res["xy"], res["grasp_z"] + 3.0, "put back")
        skip.append(res["xy"])
        await st.go_scan()
        return f"left in place ({dec.upper()}_XY not set)"
    name = "good" if dec == "box" else dec
    length, width = res.get("size", (0.0, 0.0))[:2]
    xy = free_place(spot, await lying_at(st, spot, name), length, width)
    if xy is None:
        print(f"  the {name} spot has no free place for a {length:.0f} x {width:.0f} mm part: putting it back. Clear the spot.")
        await set_down(st, res["xy"], res["grasp_z"] + 3.0, "put back")
        skip.append(res["xy"])
        await st.go_scan()
        return f"left in place ({name} spot full)"
    if not await set_down(st, xy, res["grasp_z"] + 6.0, name):
        await st.gripper.open()
        st.jaws_open = True
        await st.go_scan()
        return f"dropped on the way to the {name} spot (not placed)"
    st.placed[dec] += 1
    skip.append(np.array(xy))               # and never pick it up again from there
    await st.go_scan()
    return f"at {name} spot x={xy[0]:.0f} y={xy[1]:.0f}"


async def recover(st: Station, res: Optional[dict], skip: list) -> str:
    """After a refused move: nothing is moving. Free the gripper if it holds the part, return to the scan pose."""
    holding = (await st.gripper.is_holding_something()).is_holding_something
    if holding and res and "grasp_z" in res:
        try:      # put it back where it came from
            await set_down(st, res["xy"], res["grasp_z"] + 3.0, "recover: put back")
        except MoveFailed as e:
            if st.gpos is not None and st.gpos[2] <= res["grasp_z"] + 160.0:
                print(f"  could not put it back ({e.reason}); it is low enough to let go here")
                await st.gripper.open()
                st.jaws_open = True
            else:
                raise     # holding a part high up and stuck: stop here, a person has to sort this out
    if res and "xy" in res:
        skip.append(res["xy"])
    await st.go_scan()
    return "left on the table" if holding else "not picked"


async def watch(st: Station, log: "RunLog") -> int:
    """Continuous mode. The box is known and the arm is at the scan pose. Runs until Ctrl-C (or a cap set by
    a flag): watch with single frames, wait until the parts lie still and no hand is in view, pick, drop,
    come back, watch again. An empty table never ends it."""
    args = st.args
    if not (sys.stdin is not None and sys.stdin.isatty()):
        print("NOTE: watch mode never ends by itself. Run it from a terminal: the app sandbox kills a run at 300 s.")

    async def stop_all():
        try:
            await st.robot.stop_all()
            print("  stop_all sent")
        except Exception as e:
            print(f"  stop_all failed: {e}. Use the E-stop if the arm is moving.")

    def short(e) -> str:
        return " ".join(f"{type(e).__name__}: {getattr(e, 'message', None) or e}".split())[:160]

    def describe(o: Obj, info: dict) -> str:
        b, _ = st.locate(o, info)
        hint = " (a hand, or a part cut off by the edge of the view: slide it fully inside)" if o.kind == "rejected:edge" \
            else " (a hand, or parts touching each other: separate them)" if o.kind == "rejected:big" else ""
        return (f"{o.kind[9:]} blob {o.length_mm:.0f} x {o.width_mm:.0f} x {o.height_mm or 0:.0f} mm at x={b[0]:.0f} y={b[1]:.0f}, "
                f"not in the clean scan{hint}")

    # ---- one clean scan: what is on the table before anybody reaches in ----
    await st.open_jaws()          # same jaws in every frame from here on
    await st.confirm("clean scan for the hand rule. HANDS OFF THE TABLE and out of the camera's view (parts may stay)")
    objs, info = await st.look("scan", closeups=False)
    pose0 = pose = (info["R"], info["t"])
    raised_any, world_pts = info["raised"].copy(), [raised_world(info, st.K)]
    print(f"  learning the scene without hands: {STATIC_FRAMES} more frames ...", flush=True)
    for _ in range(STATIC_FRAMES):         # what shows up in only some frames (box edge, cable) must be known too
        _, more = await st.glance(pose0)
        raised_any |= more["raised"]
        world_pts.append(raised_world(more, st.K, limit=4000))
    static_mask = dilate(raised_any, STATIC_GROW_PX)
    static_world = np.vstack(world_pts)
    for o in objs:
        if o.kind in HAND_KINDS and dist_to_rect(st.locate(o, info)[0], st.box["bbox"]) > 60.0:
            print(f"  WARNING: the clean scan holds a {describe(o, info).split(', not in')[0]} that is not the box. It now "
                  "counts as furniture. If that was a hand or an arm, stop and start again.")
    area = view_on_table(pose0[0], pose0[1], st.K)
    print(f"\n===== watching; Ctrl-C to stop. Put parts down inside x={area[0]:.0f}..{area[1]:.0f}, y={area[2]:.0f}..{area[3]:.0f} "
          "(world mm): the camera sees that area whole. =====", flush=True)

    seen = {"world": np.zeros((0, 3))}      # what the last scan look that passed the hand rule saw standing on the table

    def close_mask(R, t0) -> np.ndarray:
        """The hand rule's mask in a close-up view: the clean scan and the last hand-free scan, seen from there,
        plus the box. Anything edge / big / tall outside it has arrived in the last couple of seconds."""
        known = np.vstack([static_world, seen["world"]])
        return project_static(known, R, t0, st.K, STATIC_GROW_PX + 4) | box_silhouette(st.box["bbox"], st.box["rim"], R, t0, st.K)

    def guard(objs_, info_, tag):
        mask = static_mask if tag == "scan" else close_mask(info_["R"], info_["t"])
        bad = hand_blobs(objs_, mask, HAND_OUTSIDE)
        if bad:
            return describe(bad[0], info_)
        if tag == "scan":
            seen["world"] = raised_world(info_, st.K)
        return None

    skips, tracks = [], []        # skips: (xy, time added). tracks: parts seen while watching, for the latency fields
    prev, away_frames = None, 2   # prev: (time, pickable xy, clean?) of the frame before; away_frames: see credit()
    delivered = attempts = errors = cam_errors = frames = 0
    blocked, rc, from_deliver = None, 0, False
    tally = {}                    # why the arm did not start, per frame, since the last status line
    t_begin = last_status = time.time()

    def near(xy, pts, radius=SKIP_RADIUS_MM) -> bool:
        return any(math.hypot(xy[0] - q[0], xy[1] - q[1]) < radius for q in pts)

    def credit(item: dict):
        """Give an item its first-seen time from the watch frames, and say whether the camera had been away
        just before that (then the part may have been waiting, and the latency is a lower bound)."""
        ref = item.get("scan_xy", item.get("xy"))
        tr = None
        if ref is not None and tracks:
            tr = min(tracks, key=lambda k: math.hypot(k["xy"][0] - ref[0], k["xy"][1] - ref[1]))
            tr = tr if near(ref, [tr["xy"]]) else None
        if tr is None:
            item.setdefault("camera_was_away", True)
            return None
        item["seen_at"], item["camera_was_away"] = tr["first_seen"], tr["away"]
        if "first_motion_at" in item:
            item["first_motion_at"] = tr.setdefault("first_motion", item["first_motion_at"])
        return tr

    async def safe_home():
        """After stop_all the arm is wherever the move got to. Straight up first if it is low, then home."""
        st.gpos = await st.transform(np.zeros(3), GRIPPER, "world")
        if st.gpos[2] < st.scan_g[2] - 5.0:
            try:
                await st.move_gripper([st.gpos[0], st.gpos[1], st.scan_g[2]], st.theta, "recover: straight up", linear=True)
            except MoveFailed as e:
                print(f"  could not go straight up ({e.reason}); going home from here")
        await st.go_home(gate=False)

    async def recover_error(e, skip_xy: list) -> bool:
        """An exception during a pick or a drop. Returns False when watch mode has to end."""
        nonlocal errors, delivered, from_deliver
        errors += 1
        text = short(e)
        print(f"\n[watch] ERROR ({errors} in a row): {text}", flush=True)
        await stop_all()
        item = st.current if st.current is not None else {}
        tr = credit(item)
        tracks[:] = [k for k in tracks if k is not tr]
        holding = False
        try:
            holding = (await st.gripper.is_holding_something()).is_holding_something
        except Exception as e2:
            print(f"  cannot read the gripper ({short(e2)})")
        still = " THE GRIPPER IS STILL HOLDING A PART: put a hand under it and open it from the CONTROL tab." if holding else ""
        if errors >= WATCH_MAX_ERRORS:
            log.add(item, f"skipped: error ({text})", decision="skip")
            print(f"[watch] {WATCH_MAX_ERRORS} errors in a row: stopping.{still}")
            return False
        try:
            await safe_home()
            if holding:           # the part is in the jaws: try the drop once more
                from_deliver = True
                result = await deliver(st, item, skip_xy)
                item["drop_at"] = st.released_at
                delivered += 0 if result.startswith("left in place") else 1
                log.add(item, f"{result}, after an error ({text})")
            else:
                if "xy" in item:
                    skip_xy.append(item["xy"])
                log.add(item, f"skipped: error ({text})", decision="skip")
                await st.open_jaws()
                await st.go_scan()
        except Exception as e2:
            await stop_all()
            log.add(item, f"skipped: error ({text}); then recovery failed ({short(e2)})", decision="skip")
            print(f"[watch] recovery failed ({short(e2)}): stopping.{still}")
            return False
        return True

    async def hold_still(why: str) -> bool:
        """A hand at the close-up: the arm stays where it is until two frames in a row are clear (True),
        or CLOSE_HAND_WAIT_S has passed (False)."""
        here = await st.cam_frame()
        mask, clear, t_hold = close_mask(*here), 0, time.time()
        print(f"[watch] holding still above the part: {why}", flush=True)
        while clear < 2 and time.time() - t_hold < CLOSE_HAND_WAIT_S:
            await asyncio.sleep(SETTLE_GAP_S)
            objs_, _ = await st.glance(here)
            clear = 0 if hand_blobs(objs_, mask, HAND_OUTSIDE) else clear + 1
        return clear >= 2

    try:
        while True:
            if args.watch_max_picks and delivered >= args.watch_max_picks:
                print(f"\n[watch] --watch-max-picks {args.watch_max_picks} reached.")
                break
            if args.watch_max_s and time.time() - t_begin >= args.watch_max_s:
                print(f"\n[watch] --watch-max-s {args.watch_max_s:.0f} reached.")
                break
            # ---- 1. watch: one frame ----
            t_loop = time.time()
            try:
                objs, info = await st.glance(pose)
                cam_errors = 0
            except Exception as e:
                cam_errors += 1
                print(f"[watch] camera frame failed ({cam_errors} in a row): {short(e)}", flush=True)
                if cam_errors >= WATCH_MAX_ERRORS:
                    rc = 10
                    break
                await asyncio.sleep(1.0)
                continue
            frames += 1
            now = info["seen_at"]
            located = [(o, tuple(float(c) for c in st.locate(o, info)[0][:2])) for o in objs]
            skips = expire_skips(skips, [xy for _, xy in located], now, SKIP_RADIUS_MM, SKIP_TTL_S)
            cands = [xy for o, xy in located if o.kind in PICK_KINDS and not near(xy, [s[0] for s in skips])]
            for tr in tracks:
                tr["missed"] += 1
            for xy in cands:
                tr = min(tracks, key=lambda k: math.hypot(k["xy"][0] - xy[0], k["xy"][1] - xy[1]), default=None)
                if tr is not None and near(xy, [tr["xy"]]):
                    tr["xy"], tr["missed"] = xy, 0
                else:
                    tracks.append(dict(xy=xy, first_seen=now, away=away_frames > 0, missed=0))
            tracks = [tr for tr in tracks if tr["missed"] < 3]
            away_frames = max(0, away_frames - 1)
            # ---- 2./3. anything to pick, and is the scene still and free of hands? ----
            bad = hand_blobs(objs, static_mask, HAND_OUTSIDE)
            blocked = describe(bad[0], info) if bad else None
            clean = bool(cands) and not bad
            ready = (clean and prev is not None and prev[2] and now - prev[0] >= SETTLE_GAP_S
                     and settled(prev[1], cands, SETTLE_TOL_MM, strict=SETTLE_STRICT, pair_mm=SKIP_RADIUS_MM))
            why = ("ready" if ready else "hand rule" if bad else "nothing pickable" if not cands
                   else "first clean frame" if (prev is None or not prev[2]) else "parts moved or changed")
            tally[why] = tally.get(why, 0) + 1
            prev = (now, cands, clean)
            if now - last_status >= WATCH_STATUS_S:
                last_status = now
                print(f"[watch {now - t_begin:5.0f} s] {frames} frames | pickable in view: {len(cands)} | skip list: {len(skips)} | "
                      f"delivered: {delivered} | errors in a row: {errors} | since last line: "
                      + ", ".join(f"{k} x{v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))
                      + f" | hand: {blocked or 'no'}", flush=True)
                tally = {}
            if not ready:
                # Idle: about one frame a second. Something in view: the next frame as soon as the settle gap allows.
                wait = (now + SETTLE_GAP_S if (cands or bad) else t_loop + WATCH_PERIOD_S) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                continue
            # ---- 4. pick and drop, as sort mode does ----
            print(f"\n[watch {now - t_begin:5.0f} s] {len(cands)} pickable part(s) lying still, no hand in view: going.", flush=True)
            attempts += 1
            st.current = None
            skip_xy = [s[0] for s in skips]
            n_skip, from_deliver, keep_track = len(skip_xy), False, False
            try:
                res = await pick_one(st, "pick", skip_xy, force_close_up=True, guard=guard)
                status = res["status"]
                if status == "none" or (status == "hand" and "xy" not in res):
                    prev = None          # nothing moved: the fused look saw no part, or a hand. Keep watching.
                    continue
                tr = credit(res)
                if status == "held":
                    from_deliver = True
                    result = await deliver(st, res, skip_xy)
                    res["drop_at"] = st.released_at
                    delivered += 0 if result.startswith("left in place") else 1
                    log.add(res, result)
                elif status == "hand":                    # seen from the close-up: no descent
                    if await hold_still(res["why"]):
                        keep_track = True                 # gone again: back up, settle again, try again
                    else:
                        skip_xy.append(res["xy"])
                        log.add(res, f"skipped: hand rule at the close-up for {CLOSE_HAND_WAIT_S:.0f} s ({res['why']})", decision="skip")
                    await st.go_scan()
                else:                                     # out / refused / lost / missed
                    if status == "missed":                # jaws closed on nothing
                        await st.gripper.open()
                        st.jaws_open = True
                    skip_xy.append(res["xy"])
                    log.add(res, f"skipped: {status}", decision="skip")
                    await st.go_scan()
                if not keep_track:
                    tracks = [k for k in tracks if k is not tr]
                errors = 0
            except Exception as e:
                if not await recover_error(e, skip_xy):
                    rc = 10
                    break
            # Spots added by a delivery (reject / hold / put back) stay skipped for as long as something lies there.
            skips += [(xy, math.inf if from_deliver else time.time()) for xy in skip_xy[n_skip:]]
            # The camera has been away: whatever shows up in the next frames may have been waiting.
            prev, away_frames = None, 2
            pose = await st.cam_frame()
            if float(np.linalg.norm(pose[1] - pose0[1])) > 3.0:
                print(f"  WARNING: back at the scan pose {np.linalg.norm(pose[1] - pose0[1]):.0f} mm from where the clean scan "
                      "was taken; the hand rule may now see the box as new and hold the arm.")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[watch] Ctrl-C: stopping.", flush=True)
        await stop_all()
    try:
        if (await st.gripper.is_holding_something()).is_holding_something:
            print("The gripper is holding a part. Put a hand under it and open it from the CONTROL tab.")
    except Exception:
        pass
    print(f"\n===== watch ended: {delivered} part(s) delivered in {attempts} attempt(s), {frames} watch frames, "
          f"{time.time() - t_begin:.0f} s =====")
    log.dump()
    return rc


def stamp(t: Optional[float]) -> str:
    return "" if t is None else time.strftime("%H:%M:%S", time.gmtime(t)) + f".{int(t * 10) % 10}Z"


class RunLog:
    """One line per item: number, class, size, position, anomaly score, decision, confidence, result, and the
    times it was first seen, the arm first moved for it, the gripper let go of it, and the item was closed.
    camera_was_away = yes: the camera was not at the scan pose just before the part was first seen (it was
    carrying another part, or the run had just started), so the part may have been lying there for a while
    and seen_to_motion_s is only a lower bound on the real latency.
    Printed as ITEM lines and appended to LOG_FILE as each item happens, when the file system allows it."""
    FIELDS = ["n", "class", "size_mm", "x", "y", "anomaly_score", "decision", "confidence", "result",
              "seen_utc", "first_motion_utc", "drop_utc", "done_utc", "seen_to_motion_s", "camera_was_away"]

    def __init__(self, path: Optional[str]):
        self.rows, self.path = [], path
        try:    # a log from an older version has other columns: leave it alone, start a new file next to it
            if path and os.path.exists(path):
                with open(path) as f:
                    if f.readline().strip() != ",".join(self.FIELDS):
                        root, ext = os.path.splitext(path)
                        self.path = root + time.strftime("_%H%M%S", time.gmtime()) + ext
                        print(f"  ({path} has other columns; logging to {self.path})")
        except OSError:
            pass

    def add(self, res: dict, result: str, decision: Optional[str] = None):
        seen, moved = res.get("seen_at"), res.get("first_motion_at")
        size = res.get("size")
        row = dict(n=len(self.rows) + 1, **{"class": res.get("kind", "?")},
                   size_mm="x".join(f"{v:.0f}" for v in size) if size else "",
                   x=f"{res['xy'][0]:.0f}" if "xy" in res else "", y=f"{res['xy'][1]:.0f}" if "xy" in res else "",
                   anomaly_score="" if res.get("anomaly") is None else f"{res['anomaly']:.2f}",
                   decision=decision or res.get("decision", "box"), confidence=f"{res.get('confidence', res.get('score', 0.0)):.2f}",
                   result=result, seen_utc=stamp(seen), first_motion_utc=stamp(moved), drop_utc=stamp(res.get("drop_at")),
                   done_utc=stamp(time.time()), seen_to_motion_s="" if not (seen and moved) else f"{moved - seen:.1f}",
                   camera_was_away={True: "yes", False: "no"}.get(res.get("camera_was_away"), ""))
        self.rows.append(row)
        print("\nITEM " + " | ".join(f"{k}={row[k]}" for k in self.FIELDS if row[k] != ""), flush=True)
        if self.path:
            try:
                new = not os.path.exists(self.path)
                with open(self.path, "a") as f:
                    if new:
                        f.write(",".join(self.FIELDS) + "\n")
                    f.write(",".join(str(row[k]).replace(",", ";") for k in self.FIELDS) + "\n")
            except OSError:
                self.path = None      # sandbox with no writable disk: the ITEM lines are the log

    def dump(self):
        print("\n===== log (csv) =====")
        print(",".join(self.FIELDS))
        for row in self.rows:
            print(",".join(str(row[k]).replace(",", ";") for k in self.FIELDS))


async def run(args) -> int:
    moving = args.mode != "look"
    if moving:
        raise ValueError('Standalone motion is disabled: use main.py --stage pick --confirm-motion for classified picking')
    args.record_hz=0  # Read-only standalone look must not toggle organizer data capture.
    print(f"pick_bolt: MODE={args.mode}" + ("  ** THE ARM WILL MOVE **" if moving else "  (no motion)"), flush=True)
    st = Station(args)
    await st.connect()
    try:
        await st.record(True)
        if moving and await st.teleop_running():
            sys.exit("teleop is running. Stop it (T in the capture webapp) and move the leader arm away first.")
        if moving:
            held = await st.gripper.is_holding_something()
            if held.is_holding_something:
                print("The gripper is already holding something (from the last pick?). Empty it first: CONTROL tab -> "
                      "gripper -> Open, with a hand under it. Not moving.")
                return 7

        if args.mode in ("look", "hover", "pick"):
            if moving and args.home:
                await st.go_home()
                await st.go_scan(gate=False)
            res = await pick_one(st, args.mode, [])
            if res["status"] == "held":
                print("Part is in the gripper. The gripper stays closed.")
            return {"none": 3, "out": 4, "lost": 5, "refused": 6, "missed": 2}.get(res["status"], 0)

        # ---- box / sort ----
        args.index = None                  # sort chooses for itself, biggest part first
        await st.go_home()
        if args.good_xy is not None:
            # Good parts are set down at GOOD_XY: no box to find, nothing dropped.
            # ponytail: the spot stands in as a wall-less box, so every box rule keeps working unchanged (parts are
            # not picked from it, what lies there is not taken for a hand, reject / hold spots keep their distance).
            # Give the place area a type of its own if it ever needs rules a box does not have.
            gx, gy = (float(c) for c in args.good_xy)
            area = place_area(args.good_xy)
            if not all(reachable([x, y, PLACE_CARRY_MIN_Z]) for x, y in place_grid(args.good_xy)):
                print(f"GOOD_XY {[gx, gy]} (or the far end of its places) is out of reach at PLACE_CARRY_MIN_Z. "
                      "Move it closer to the arm and run again.")
                return 9
            st.box = dict(drop=np.array([gx, gy]), rim=TABLE_Z_EXPECTED, drop_z=float(st.scan_g[2]), bbox=area)
            print(f"\nGood parts are set down from x={gx:.0f} y={gy:.0f} (GOOD_XY), on free places within "
                  f"x={area[0]:.0f}..{area[1]:.0f} y={area[2]:.0f}..{area[3]:.0f}. No box is used.")
        elif not await find_box(st):
            return 8
        for name, spot in (("REJECT_XY", args.reject_xy), ("HOLD_XY", args.hold_xy)):
            if spot is not None and dist_to_rect(spot, st.box["bbox"]) < SPOT_KEEPOUT_MM:
                print(f"{name} {list(spot)} is within {SPOT_KEEPOUT_MM:.0f} mm of the box. Move it and run again.")
                return 9
            if spot is not None and not all(reachable([x, y, PLACE_CARRY_MIN_Z]) for x, y in place_grid(spot)):
                print(f"{name} {list(spot)} (or the far end of its places) is out of reach at PLACE_CARRY_MIN_Z.")
                return 9
        if args.mode == "box":
            print('\nBox only. Next step: MODE = "sort".')
            return 0
        await st.go_scan()
        if args.mode == "watch":
            return await watch(st, RunLog(args.log_file))
        log, skip, done, tries, last = RunLog(args.log_file), [], 0, 0, {}
        while done < args.max_picks and tries < args.max_picks + 4:
            if st.elapsed() > args.time_budget_s:
                print(f"\nTime budget ({args.time_budget_s:.0f} s) reached. Run again for the rest.")
                break
            tries += 1
            st.current = None
            try:
                res = last = await pick_one(st, "pick", skip)
                if res["status"] == "none":
                    break
                if res["status"] == "held":
                    result = await deliver(st, res, skip)
                    res["drop_at"] = st.released_at
                    done += 0 if result.startswith("left in place") else 1
                    log.add(res, result)
                    continue
                if res["status"] == "missed":         # jaws closed on nothing
                    await st.gripper.open()
                    st.jaws_open = True
                skip.append(res["xy"])                # do not try the same spot again in this run
                log.add(res, f"skipped: {res['status']}", decision="skip")
                await st.go_scan()
            except MoveFailed as e:
                # One refused move must not end the run: free the gripper, skip this part, carry on.
                print(f"\nMOVE REFUSED - {e.what}: {e.reason}")
                res = st.current or {}
                result = await recover(st, res, skip)
                log.add(res, f"skipped: planner refused '{e.what}' ({result})", decision="skip")
        for kind, L, W_, H_, u, v in last.get("left", []):     # what is still there and why it was never tried
            log.add(dict(kind=kind, size=(L, W_, H_)), f"not attempted: not pickable (pixel {u:.0f},{v:.0f})", decision="none")
        print(f"\n===== sort finished: {done} part(s) delivered, {st.elapsed():.0f} s =====")
        log.dump()
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError, Exception):
        if st.moved:
            try:
                await st.robot.stop_all()
                print("\nstop_all sent")
            except Exception as e:
                print(f"\nstop_all failed: {e}. Use the E-stop if the arm is moving.")
        raise
    finally:
        try:
            await st.record(False)
        except Exception:
            pass
        await st.close()


def parse_args(argv=None):
    env_addr = os.environ.get("VIAM_MACHINE_ADDRESS") or os.environ.get("VIAM_ADDRESS") or DEFAULT_ADDRESS
    ap = argparse.ArgumentParser(description="Pick bolts off the table with the wrist camera and drop them in a box.")
    ap.add_argument("--mode", choices=["look", "hover", "pick", "box", "sort", "watch"], default=MODE)
    ap.add_argument("--address", default=env_addr)
    ap.add_argument("--yes", action="store_true", help="no prompts and no pauses before moves")
    ap.add_argument("--no-home", dest="home", action="store_false", default=GO_HOME_FIRST,
                    help="look/hover/pick: scan from the current pose instead of home")
    ap.add_argument("--index", type=int, default=PICK_INDEX, help="blob number to go for instead of auto-choosing")
    ap.add_argument("--close-up", choices=["auto", "always"], default=CLOSE_UP)
    ap.add_argument("--jaw-axis", choices=["x", "y"], default=JAW_AXIS)
    ap.add_argument("--tip-offset-mm", type=float, default=TIP_OFFSET_MM)
    ap.add_argument("--tip-clearance-mm", type=float, default=TIP_CLEARANCE_MM)
    ap.add_argument("--pause-s", type=float, default=PAUSE_S, help="pause before gated moves when there is no terminal")
    ap.add_argument("--frames", type=int, default=FRAMES, help="point clouds fused per look")
    ap.add_argument("--record-hz", type=float, default=RECORD_HZ, help="record cameras/arm/gripper to Viam data; 0 = off")
    ap.add_argument("--box-hint", type=float, nargs=2, default=BOX_HINT_XY, metavar=("X", "Y"))
    ap.add_argument("--good-xy", type=float, nargs=2, default=GOOD_XY, metavar=("X", "Y"),
                    help="set good parts down at this table spot instead of dropping them into a box")
    ap.add_argument("--reject-xy", type=float, nargs=2, default=REJECT_XY, metavar=("X", "Y"))
    ap.add_argument("--hold-xy", type=float, nargs=2, default=HOLD_XY, metavar=("X", "Y"))
    ap.add_argument("--anomaly-service", default=ANOMALY_SERVICE)
    ap.add_argument("--log-file", default=LOG_FILE)
    ap.add_argument("--max-picks", type=int, default=MAX_PICKS)
    ap.add_argument("--time-budget-s", type=float, default=TIME_BUDGET_S)
    ap.add_argument("--watch-max-picks", type=int, default=WATCH_MAX_PICKS, help="watch: stop after this many parts; 0 = no cap")
    ap.add_argument("--watch-max-s", type=float, default=WATCH_MAX_S, help="watch: stop after this many seconds; 0 = no cap")
    ap.add_argument("--force", action="store_true", help="ignore the table-height sanity check")
    ap.add_argument("--debug-dir", default=".")
    ap.add_argument("--save-raw", action="store_true", help="also save the depth frames as .npz for offline tuning")
    return ap.parse_known_args(argv)[0]   # tolerate whatever arguments a sandbox passes


async def main():
    """Entry point. The app sandbox may already run an event loop, so this stays awaitable."""
    return await run(parse_args(sys.argv[1:]))


def self_check():
    """Offline check of the size-driven grasp rule, no robot: python3 pick_bolt.py --self-check
    Expected values hold for the size settings as shipped; the fingertip clearance is pinned to 2 mm here."""
    # (length, width, height) -> (across, upright, fingertip mm above the table), or None = the jaws cannot take it.
    cases = [((83, 31, 22), (True, False, 2.0)),      # cap bolt lying    } the parts measured on this table: these
             ((69, 21, 12), (True, False, 2.0)),      # lag screw lying   } must come out as the per-kind rules
             ((37, 29, 41), (False, True, 27.0)),     # flange bolt standing } gave them before grasp_plan
             ((27, 25, 14), (False, False, 2.0)),     # nut
             ((20, 18, 6.5), (False, False, 1.5)),    # thin nut
             ((83, 31, 38), (True, False, 13.0)),     # tall lying part: fingertips rise so the pad stays on it
             ((60, 55, 12), (False, False, 2.0)),     # not a fastener: low block that fits either way
             ((80, 60, 12), (True, False, 2.0)),      # too long one way: taken across its narrow side
             ((80, 75, 12), None),                    # too wide both ways
             ((100, 60, 50), None)]                   # tall, and longer than an upright part may read
    for dims, want in cases:
        got = grasp_plan(*dims, tip_clearance_mm=2.0)
        assert got == want, f"grasp_plan{dims} = {got}, expected {want}"
    assert grasp_plan(100, 30, 22, upright=True) == (False, True, STANDING_TIP_MM)   # an earlier look said upright
    # Letting go over the box (rim, floor, room to the walls) -> fingertip height. Carrying height is rim + 90.
    assert release_tip_z(200, 5, 70) == 155      # open jaws fit: fingertips 45 below the rim
    assert release_tip_z(200, 5, 85) == 80       # gripper body fits too: 120 below the rim
    assert release_tip_z(200, 5, 35) == 210      # tight box: stay 10 above the rim
    assert release_tip_z(100, 5, 85) == 65       # shallow box: the floor clearance wins
    assert release_tip_z(200, -30, 85) == 80 and release_tip_z(90, -30, 85) == 60   # a floor read below the table is not believed

    # With GOOD_XY set a good part is set down at the spot (lowered, then released), never dropped, and never on
    # top of what already lies there. Fake station: its camera sits at the gripper, 440 mm up, looking down.
    K = Intrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)
    down = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])      # image down = toward the arm's base

    class Fake:
        def __init__(self, lying=()):
            self.args = argparse.Namespace(good_xy=[300.0, 250.0], reject_xy=[300.0, -250.0], hold_xy=None)
            self.placed, self.moves, self.gripper = {"box": 0, "reject": 0, "hold": 0}, [], self
            self.K, self.lying = K, list(lying)            # lying: where parts were let go
            self.gpos, self.scan_g = np.array([400.0, 0.0, 350.0]), np.array([350.0, 0.0, 350.0])

        async def move_gripper(self, xyz, theta, what, linear=False, gate=False):
            self.gpos = np.array(xyz, float)

        async def open(self):
            self.moves.append([round(float(c)) for c in self.gpos])
            self.lying.append(self.gpos[:2].copy())

        async def is_holding_something(self):
            return argparse.Namespace(is_holding_something=True, meta=None)

        async def confirm(self, what, gate=True):
            pass

        async def go_scan(self, gate=False):
            pass

        def gripper_xyz_for_camera(self, cam_xyz):
            return np.asarray(cam_xyz, float)

        async def look(self, tag, closeups=True):       # a 20 mm cube wherever a part was let go
            depth = np.full((480, 640), 440.0)
            for x, y in self.lying:
                v, u = int(240 - (x - self.gpos[0]) * 600 / 440), int(320 - (y - self.gpos[1]) * 600 / 440)
                depth[max(v - 14, 0):max(v + 14, 0), max(u - 14, 0):max(u + 14, 0)] = 420.0
            info = detect_depth(depth, K)[1]
            info["R"], info["t"] = down, np.array([self.gpos[0], self.gpos[1], 440.0])
            return [], info
    st, skip = Fake(), []
    part = dict(decision="box", xy=np.array([400.0, 0.0]), grasp_z=69.0, size=(20.0, 20.0, 20.0))
    for _ in range(2):
        result = asyncio.run(deliver(st, part, skip))
    again = Fake(st.lying)                  # the next run remembers nothing; the parts are still on the table
    asyncio.run(deliver(again, part, skip))
    spots = st.moves + again.moves
    assert spots[0] == [300, 250, 75], spots      # the first place, opened 6 mm above the height it was grasped at
    assert min(math.dist(p[:2], q[:2]) for i, p in enumerate(spots) for q in spots[:i]) >= 20 + PLACE_GAP_MM, spots
    assert result.startswith("at good spot") and len(skip) == 3, result
    asyncio.run(deliver(again, dict(part, decision="reject"), skip))
    assert again.moves[-1] == [300, -250, 75], again.moves         # rejects go to the other end of the table
    assert place_grid((300.0, -250.0))[PLACE_COLS].tolist() == [300.0, -250.0 - PLACE_PITCH_MM]   # rows grow away
    full = place_grid((300.0, 250.0))
    assert free_place((300.0, 250.0), full, 20.0, 20.0) is None
    beside = np.array([300.0, 250.0]) + 30.0 * np.array([-250.0, 300.0]) / math.hypot(300.0, 250.0)
    assert free_place((300.0, 250.0), [beside], 20.0, 20.0).tolist() != [300.0, 250.0]   # the opening jaws would push it

    class Flat(Fake):        # a nut read 4 mm high: no raised pixels, only a blob, right under the camera
        async def look(self, tag, closeups=True):
            info = detect_depth(np.full((480, 640), 440.0), K)[1]
            info["R"], info["t"] = down, np.array([self.gpos[0], self.gpos[1], 440.0])
            ys, xs = np.mgrid[226:254, 306:334]
            return [argparse.Namespace(xs=xs.ravel(), ys=ys.ravel(), height_mm=4.0)], info
    flat = Flat()
    taken = asyncio.run(lying_at(flat, (300.0, 250.0), "good"))
    assert len(taken) and np.hypot(*(taken - place_grid((300.0, 250.0)).mean(axis=0)).T).min() < 15, taken

    class Tape(Flat):        # tape marking the spot: 1 mm high, never a part
        async def look(self, tag, closeups=True):
            objs, info = await Flat.look(self, tag, closeups)
            objs[0].height_mm = 1.0
            return objs, info
    assert not len(asyncio.run(lying_at(Tape(), (300.0, 250.0), "good")))

    class Far(Fake):         # a place beyond reach at carrying height: approached lower, left in one slanted line
        def __init__(self):
            super().__init__()
            self.args.good_xy, self.scan_g, self.path = [370.0, 465.0], np.array([336.0, 14.0, 353.0]), []

        async def move_gripper(self, xyz, theta, what, linear=False, gate=False):
            self.path.append((what, np.array(xyz, float)))
            self.gpos = np.array(xyz, float)
    far = Far()
    asyncio.run(deliver(far, dict(part, size=(80.0, 25.0, 20.0)), []))
    setting = [p for what, p in far.path if not what.startswith("good: camera")]      # the Fake camera sits at the gripper
    assert far.moves == [[370, 465, 75]] and all(reachable(p) for p in setting), far.path
    assert setting[0][2] < far.scan_g[2] and setting[-1].tolist() == far.scan_g.tolist(), far.path

    # A standing 80 mm bolt seen from the scan, camera 440 mm straight above x=400 y=0: its top face 360 mm away.
    # The batch target position is 15 mm off the top. pick_one aims at the top, comes down around it to
    # STANDING_TIP_MM from TOP_CLEAR_MM above it, and never takes a close-up.
    up_depth = np.full((480, 640), 440.0)
    up_depth[262:276, 285:299] = 360.0
    class Upright(Station):
        def __init__(self):
            self.args = argparse.Namespace(index=None, force=False, close_up="auto", jaw_axis="y",
                                           tip_offset_mm=TIP_OFFSET_MM, tip_clearance_mm=2.0)
            self.K, self.t_gc, self.box, self.vision, self.current, self.moves, self.gripper = K, np.zeros(3), None, None, None, [], self
            self.inspection_handler, self.looks = None, []

        async def look(self, tag, closeups=True):
            self.looks.append(tag)
            objs, info = detect_depth(up_depth, K)
            info.update(R=down, t=np.array([400.0, 0.0, 440.0]), seen_at=0.0, table_z=0.0)
            return objs, info

        async def move_gripper(self, xyz, theta, what, linear=False, gate=False):
            self.moves.append((what, [float(c) for c in xyz]))

        async def open_jaws(self):
            pass

        async def confirm(self, what, gate=True):
            pass

        async def grab(self):
            return True

        async def is_holding_something(self):
            return argparse.Namespace(is_holding_something=True, meta=None)
    arm = Upright()
    with open(os.devnull, "w") as quiet:
        out_, sys.stdout = sys.stdout, quiet
        try:
            res = asyncio.run(pick_one(arm, "pick", [], target_world_xy=np.array([400.0 - 15.0, 0.0]), target_tolerance_mm=15.0,
                                       roi_target=lambda info: (_ for _ in ()).throw(ValueError("Insufficient ROI foreground"))))
        finally:
            sys.stdout = out_
    face = np.array([400.0 - (268.5 - 240) / 600 * 360, 0.0 - (291.5 - 320) / 600 * 360])     # world xy of the top face
    moves = dict(arm.moves)
    assert res["status"] == "held" and res["kind"] == "bolt-standing" and math.dist(res["xy"], face) < 3, res
    assert arm.looks == ["scan"], arm.looks                                                       # no close-up
    assert moves["pre-grasp above the standing part"][2] - TIP_OFFSET_MM >= 80 + TOP_CLEAR_MM - 3, moves
    assert abs(moves["descend around the standing part"][2] - TIP_OFFSET_MM - STANDING_TIP_MM) < 1, moves
    # The same bolt with a tape flag on its head, as high as the head, sticking out along image u (world -y; the
    # station's 14 Nov miss): the jaws close along the flag, so they slide the shank to the middle.
    up_depth[266:272, 299:333] = 360.0
    flagged = Upright()
    with open(os.devnull, "w") as quiet:
        out_, sys.stdout = sys.stdout, quiet
        try:
            res = asyncio.run(pick_one(flagged, "pick", [], target_world_xy=np.array([400.0 - 15.0, 0.0]), target_tolerance_mm=15.0,
                                       roi_target=lambda info: (_ for _ in ()).throw(ValueError("Insufficient ROI foreground"))))
        finally:
            sys.stdout = out_
    jaws = np.array([math.sin(math.radians(res["theta"])), math.cos(math.radians(res["theta"]))])   # gripper y, jaws close along it
    assert res["status"] == "held" and abs(jaws[1]) > 0.99, res
    # A thin standing screw whose top gives the scan no depth: the home capture's height still makes it a standing pick.
    up_depth[:] = 440.0
    thin = Upright()
    with open(os.devnull, "w") as quiet:
        out_, sys.stdout = sys.stdout, quiet
        try:
            res = asyncio.run(pick_one(thin, "pick", [], target_world_xy=np.array([390.0, 10.0]), target_tolerance_mm=15.0,
                                       roi_target=lambda info: (_ for _ in ()).throw(ValueError("Insufficient ROI foreground")),
                                       target_height_mm=72.0))
        finally:
            sys.stdout = out_
    assert res["status"] == "held" and res["kind"] == "bolt-standing" and res["xy"].tolist() == [390.0, 10.0], res
    # End to end through the detector: a 60 x 55 x 12 mm block on a flat table, camera 440 mm straight above.
    depth = np.full((480, 640), 440.0)
    depth[200:275, 280:362] = 428.0                   # 82 x 75 px at 428 mm = 58 x 53 mm
    objs, _ = detect_depth(depth, K)
    found = [(o.kind, round(o.length_mm), round(o.width_mm), round(o.height_mm)) for o in objs]
    assert len(objs) == 1 and objs[0].kind == "object" and abs(objs[0].length_mm - 58) < 4 \
        and abs(objs[0].width_mm - 53) < 4 and abs(objs[0].height_mm - 12) < 1, f"detector saw {found}"
    print("self-check passed:", found)


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
        sys.exit(0)
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
