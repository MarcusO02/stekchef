import os, glob, time, json, cv2, numpy as np
import functools, builtins

print = functools.partial(builtins.print, flush=True)

# -------------------- Config --------------------
BASEPATH = os.getcwd()
# Use the same folder as your other stream, or point to your capture path
FOLDER = r'C:\Users\marcu\source\repos\ConsoleApp1\ConsoleApp1\bin\Debug\frames_out'
POLL_SECONDS = 0.10
DEPTH_CLIP_MM = 4000

# Kinect-ish base intrinsics (scaled to frame size at runtime)
KINECT_BASE_W, KINECT_BASE_H = 640, 480
KINECT_BASE_FX = 525.0
KINECT_BASE_FY = 525.0
KINECT_BASE_CX = 320.0
KINECT_BASE_CY = 240.0

# Plane & blob settings
SUBSAMPLE_STEP = 4
RANSAC_ITERS = 600
RANSAC_INLIER_MM = 12.0
HEIGHT_THRESH_MM = 3.0         # height above plane for "on table"
MORPH_KERNEL = (3, 3)
REFIT_PLANE_EVERY_N = 30

# Pan candidates: size filter (on cropped image)
PAN_MIN_BLOB_AREA = 500       # px
PAN_MAX_BLOB_AREA = 200000    # px

# Crop early (like in your working finder)
CROP_TOP_Y_FRAC   = 0.25
CROP_SIDE_X_FRAC  = 0.05

# -------------------- Small utilities --------------------
def idle_ms(ms, show=False):
    """Sleep or pump GUI depending on show flag."""
    if show:
        cv2.waitKey(max(1, int(ms)))
    else:
        time.sleep(ms / 1000.0)

def _read_text(path):
    for _ in range(20):
        try:
            with open(path, "r") as f:
                return f.read()
        except:
            time.sleep(0.05)
    raise IOError(f"Could not read {path}")

def _read_binary(path):
    prev = -1
    for _ in range(20):
        try:
            size1 = os.path.getsize(path)
            if size1 != prev and prev != -1:
                prev = size1
                time.sleep(0.05)
                continue
            with open(path, "rb") as f:
                data = f.read()
            if len(data) == size1 and size1 > 0:
                return data
        except:
            pass
        prev = size1
        time.sleep(0.05)
    raise IOError(f"Could not reliably read {path}")

def loadExtrinsic():
    filePath = os.path.join(BASEPATH, "extrinsics_results.json")
    with open(filePath, "r") as f:
        extrinsics = json.load(f)
    ent = extrinsics.get("image_0.png", None)
    if ent is None:
        first_key = next(iter(extrinsics.keys()))
        ent = extrinsics[first_key]
    rvec = np.array(ent["rvec"], dtype=np.float32).reshape(3, 1)
    tvec = np.array(ent["tvec"], dtype=np.float32).reshape(3, 1)
    return rvec, tvec

def load_aligned_depth_pair(base):
    shape_txt = base + ".shape.txt"
    raw_path  = base + ".depth16le"
    if not os.path.exists(shape_txt) or not os.path.exists(raw_path):
        raise IOError(f"Missing depth files for base {base}")
    meta = _read_text(shape_txt).strip().split()
    if len(meta) < 2:
        raise ValueError(f"Shape file {shape_txt} has unexpected content: {meta}")
    w, h = map(int, meta[:2])
    depth = np.frombuffer(_read_binary(raw_path), dtype=np.uint16).reshape(h, w)
    ts = os.path.basename(base).split("_")[-1]
    rgb_path = os.path.join(os.path.dirname(base), f"rgb_{ts}.png")
    return depth, (rgb_path if os.path.exists(rgb_path) else None)

def intrinsics_for_shape(w, h):
    sx, sy = w / KINECT_BASE_W, h / KINECT_BASE_H
    return (KINECT_BASE_FX * sx,
            KINECT_BASE_FY * sy,
            KINECT_BASE_CX * sx,
            KINECT_BASE_CY * sy)

# -------------------- Geometry --------------------
def backproject(depth_mm, fx, fy, cx, cy):
    h, w = depth_mm.shape
    Z = depth_mm.astype(np.float32) / 1000.0
    u = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    v = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z

def ransac_floor_plane(depth_mm, fx, fy, cx, cy):
    Zm = depth_mm[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP].astype(np.float32) / 1000.0
    vv, uu = np.indices(Zm.shape, dtype=np.float32)
    uu *= SUBSAMPLE_STEP; vv *= SUBSAMPLE_STEP
    valid = (Zm > 0) & (depth_mm[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP] < DEPTH_CLIP_MM)
    if valid.sum() < 3:
        return None
    uu, vv, Zs = uu[valid], vv[valid], Zm[valid]
    Xs = (uu - cx) * Zs / fx
    Ys = (vv - cy) * Zs / fy
    P = np.column_stack((Xs, Ys, Zs))

    best_inliers = -1
    best_plane = None
    thr = RANSAC_INLIER_MM / 1000.0
    rng = np.random.default_rng()
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(P.shape[0], 3, replace=False)
        p1, p2, p3 = P[idx]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -np.dot(n, p1)
        dist = np.abs(P @ n + d)
        inliers = np.count_nonzero(dist < thr)
        if inliers > best_inliers:
            best_inliers, best_plane = inliers, (float(n[0]), float(n[1]), float(n[2]), float(d))
    return best_plane

def height_above_plane_mm(X, Y, Z, plane):
    a, b, c, d = plane
    return np.abs(a * X + b * Y + c * Z + d) * 1000.0

# -------------------- Color: blackMask (pan) --------------------
def blackMask(bgr_img, blur_ksize=3):
    """
    Detects very dark regions in HSV (low V).
    Returns uint8 mask (0/255); expects aligned/cropped RGB.
    """
    if bgr_img is None:
        return None
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,   0,   0],  dtype=np.uint8)
    upper = np.array([180, 255, 60], dtype=np.uint8)  # V<=60 → very dark
    mask = cv2.inRange(hsv, lower, upper)
    if blur_ksize and blur_ksize > 1:
        mask = cv2.medianBlur(mask, blur_ksize)
    return mask

# -------------------- Cropping --------------------
def apply_crop(depth, rgb, fx, fy, cx, cy):
    """
    Crop 25% from top and 5% from each side; adjust cx, cy accordingly.
    """
    h, w = depth.shape
    x0 = int(round(CROP_SIDE_X_FRAC * w))
    x1 = int(round((1.0 - CROP_SIDE_X_FRAC) * w))
    y0 = int(round(CROP_TOP_Y_FRAC * h))
    y1 = h

    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 1, min(y1, h))

    depth_c = depth[y0:y1, x0:x1]
    rgb_c = rgb[y0:y1, x0:x1] if rgb is not None else None

    cx_c = cx - x0
    cy_c = cy - y0
    return depth_c, rgb_c, fx, fy, cx_c, cy_c

# -------------------- Main: pan position stream --------------------
def getPanPositions(show=False):
    """
    Generator yielding (x, y, z) in your robot/IK frame for the pan location.
    Uses: height-above-plane ∧ valid-depth ∧ blackMask blob (largest by area).
    """
    os.makedirs(FOLDER, exist_ok=True)
    print("Pan tracker initialized — streaming pan positions...")

    # Extrinsics camera->world (same as your other stream)
    rvec, tvec = loadExtrinsic()
    R, _ = cv2.Rodrigues(rvec)

    plane_model = None
    frames_since_plane = REFIT_PLANE_EVERY_N
    last_base = None
    frame_idx = 0

    while True:
        depth_files = [
            f for f in glob.glob(os.path.join(FOLDER, "depth_aligned_*.depth16le"))
            if os.path.exists(f[:-len(".depth16le")] + ".shape.txt")
        ]
        if not depth_files:
            idle_ms(POLL_SECONDS * 1000, show)
            continue

        latest_depth_file = max(depth_files, key=os.path.getmtime)
        base = latest_depth_file[:-len(".depth16le")]
        if base == last_base:
            idle_ms(POLL_SECONDS * 1000, show)
            continue

        try:
            depth, rgb_path = load_aligned_depth_pair(base)
        except Exception as e:
            print("Load failed:", e)
            idle_ms(POLL_SECONDS * 1000, show)
            continue

        frame_idx += 1
        last_base = base
        print(f"\n--- Pan Frame {frame_idx} ---")

        # Intrinsics for the ORIGINAL shape
        h0, w0 = depth.shape
        fx, fy, cx, cy = intrinsics_for_shape(w0, h0)
        rgb_img = cv2.imread(rgb_path) if rgb_path is not None else None

        # Crop depth+rgb+intrinsics (as in your working code)
        depth, rgb_img, fx, fy, cx, cy = apply_crop(depth, rgb_img, fx, fy, cx, cy)

        # Fit/refresh plane on the CROPPED depth
        if plane_model is None or frames_since_plane >= REFIT_PLANE_EVERY_N:
            plane_model = ransac_floor_plane(depth, fx, fy, cx, cy)
            frames_since_plane = 0
            if plane_model is None:
                print("No plane found.")
                # still let UI breathe
                idle_ms(1, show)
                continue
        else:
            frames_since_plane += 1

        # Backproject and height-above-plane
        X, Y, Z = backproject(depth, fx, fy, cx, cy)
        valid = (depth > 0) & (depth < DEPTH_CLIP_MM)
        H_mm = height_above_plane_mm(X, Y, Z, plane_model)

        # Pan mask: height ∧ valid ∧ black
        mask_black = blackMask(rgb_img) if rgb_img is not None else None
        if mask_black is None:
            mask_black = np.zeros_like(depth, dtype=np.uint8)

        pan_mask = (H_mm > HEIGHT_THRESH_MM) & valid & (mask_black > 0)
        pan_u8 = (pan_mask.astype(np.uint8) * 255)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
        pan_u8 = cv2.morphologyEx(pan_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        pan_u8 = cv2.morphologyEx(pan_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Largest connected component in area limits
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(pan_u8, connectivity=8)
        best_idx = -1
        best_area = -1
        for i in range(1, num):
            _, _, _, _, area = stats[i]
            if area < PAN_MIN_BLOB_AREA or area > PAN_MAX_BLOB_AREA:
                continue
            if area > best_area:
                best_area = area
                best_idx = i

        pan_xyz_robot = None
        if best_idx > 0:
            mask_i = (labels == best_idx)

            # Robust median 3D
            Z_blob = Z[mask_i]
            Z_blob = Z_blob[(Z_blob > 0) & np.isfinite(Z_blob)]
            if Z_blob.size > 0:
                z_m = float(np.median(Z_blob))

                X_blob, Y_blob = X[mask_i], Y[mask_i]
                X_blob = X_blob[np.isfinite(X_blob)]
                Y_blob = Y_blob[np.isfinite(Y_blob)]
                if X_blob.size > 0 and Y_blob.size > 0:
                    x_cam = float(np.median(X_blob))
                    y_cam = float(np.median(Y_blob))

                    X_cam = np.array([x_cam, y_cam, z_m], dtype=np.float64).reshape(3, 1)
                    # world: X_world = R^T (X_cam - tvec)
                    X_world = R.T @ (X_cam - tvec.reshape(3, 1))
                    Xw = X_world.ravel()

                    # Same handedness tweak as your object stream
                    # (keep z as-is; you can add a small +clearance later in the approach path)
                    pan_xyz_robot = (-float(Xw[0]) - 0.03, float(Xw[1]), float(Xw[2]))

                    # Show overlay on RGB (blue box + label)
                    if show and (rgb_img is not None):
                        try:
                            x, y, ww, hh, _ = stats[best_idx]
                            cx_px, cy_px = int(round(centroids[best_idx][0])), int(round(centroids[best_idx][1]))
                            disp = rgb_img.copy()
                            cv2.rectangle(disp, (x, y), (x + ww, y + hh), (255, 0, 0), 2)
                            cv2.circle(disp, (cx_px, cy_px), 6, (255, 0, 0), -1)
                            txt = f"pan world: x={Xw[0]:.3f} y={Xw[1]:.3f} z={Xw[2]:.3f} m"
                            cv2.putText(disp, txt, (max(0, x), max(12, y - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 0), 1, cv2.LINE_AA)
                            cv2.imshow("Pan finder (RGB)", disp)
                            idle_ms(1, show)
                        except Exception as e:
                            print("Display error:", e)

        if pan_xyz_robot is not None:
            yield pan_xyz_robot

        idle_ms(10, show)
