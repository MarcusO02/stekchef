# perception_core.py
import os, time, json, numpy as np, cv2

# Intrinsics (base, scaled per frame)
KINECT_BASE_W, KINECT_BASE_H = 640, 480
KINECT_BASE_FX = 525.0
KINECT_BASE_FY = 525.0
KINECT_BASE_CX = 320.0
KINECT_BASE_CY = 240.0

SUBSAMPLE_STEP = 4
RANSAC_ITERS = 600
RANSAC_INLIER_MM = 12.0
DEPTH_CLIP_MM = 4000

def idle_ms(ms, show=False):
    if show: cv2.waitKey(max(1, int(ms)))
    else:    time.sleep(ms/1000.0)

def loadExtrinsic(basepath):
    path = os.path.join(basepath, "extrinsics_results.json")
    with open(path, "r") as f:
        extr = json.load(f)
    ent = extr.get("image_0.png") or extr[next(iter(extr.keys()))]
    rvec = np.array(ent["rvec"], dtype=np.float32).reshape(3,1)
    tvec = np.array(ent["tvec"], dtype=np.float32).reshape(3,1)
    return rvec, tvec

def _read_text(path):
    for _ in range(20):
        try:    return open(path, "r").read()
        except: time.sleep(0.05)
    raise IOError(f"Could not read {path}")

def _read_binary(path):
    prev = -1
    for _ in range(20):
        try:
            size1 = os.path.getsize(path)
            if size1 != prev and prev != -1:
                prev = size1; time.sleep(0.05); continue
            data = open(path, "rb").read()
            if len(data) == size1 and size1 > 0:
                return data
        except: pass
        prev = size1; time.sleep(0.05)
    raise IOError(f"Could not reliably read {path}")

def load_aligned_depth_pair(base):
    shape_txt = base + ".shape.txt"
    raw_path  = base + ".depth16le"
    if not (os.path.exists(shape_txt) and os.path.exists(raw_path)):
        raise IOError(f"Missing depth files for base {base}")
    w, h = map(int, _read_text(shape_txt).strip().split()[:2])
    depth = np.frombuffer(_read_binary(raw_path), dtype=np.uint16).reshape(h, w)
    ts = os.path.basename(base).split("_")[-1]
    rgb_path = os.path.join(os.path.dirname(base), f"rgb_{ts}.png")
    return depth, (rgb_path if os.path.exists(rgb_path) else None)

def intrinsics_for_shape(w, h):
    sx, sy = w / KINECT_BASE_W, h / KINECT_BASE_H
    return KINECT_BASE_FX*sx, KINECT_BASE_FY*sy, KINECT_BASE_CX*sx, KINECT_BASE_CY*sy

def backproject(depth_mm, fx, fy, cx, cy):
    h, w = depth_mm.shape
    Z = depth_mm.astype(np.float32)/1000.0
    u = np.tile(np.arange(w, dtype=np.float32), (h,1))
    v = np.tile(np.arange(h, dtype=np.float32).reshape(-1,1), (1,w))
    X = (u - cx)*Z/fx
    Y = (v - cy)*Z/fy
    return X, Y, Z

def ransac_floor_plane(depth_mm, fx, fy, cx, cy):
    h, w = depth_mm.shape
    Zm = depth_mm[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP].astype(np.float32)/1000.0
    vv, uu = np.indices(Zm.shape, dtype=np.float32)
    uu *= SUBSAMPLE_STEP; vv *= SUBSAMPLE_STEP
    valid = (Zm>0) & (depth_mm[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP] < DEPTH_CLIP_MM)
    if valid.sum() < 3:
        return None
    uu, vv, Zs = uu[valid], vv[valid], Zm[valid]
    Xs = (uu - cx)*Zs/fx; Ys = (vv - cy)*Zs/fy
    P = np.column_stack((Xs, Ys, Zs))
    best_inliers=-1; best_plane=None; thr=RANSAC_INLIER_MM/1000
    rng=np.random.default_rng()
    for _ in range(RANSAC_ITERS):
        idx=rng.choice(P.shape[0],3,replace=False)
        p1,p2,p3 = P[idx]
        n = np.cross(p2-p1,p3-p1); norm = np.linalg.norm(n)
        if norm<1e-6: continue
        n /= norm; d = -np.dot(n,p1)
        dist = np.abs(P@ n + d)
        inliers = np.count_nonzero(dist<thr)
        if inliers>best_inliers:
            best_inliers,best_plane=inliers,(float(n[0]),float(n[1]),float(n[2]),float(d))
    return best_plane

def plane_z_at_xy(plane, x, y):
    a, b, c, d = plane
    if abs(c) < 1e-9: return None
    return -(a*x + b*y + d)/c
