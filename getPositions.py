# get_positions.py
import os, glob, time, cv2, numpy as np
import functools, builtins

from colorFilters import redMask
from approach_path import plan_adaptive        # for overlay only
from overlays import project_world_points
from approach_profile import render_approach_profile

# Shared core utilities
from perception import (
    idle_ms, loadExtrinsic, load_aligned_depth_pair,
    intrinsics_for_shape, backproject, ransac_floor_plane
)

print = functools.partial(builtins.print, flush=True)

# --- Config ---
BASEPATH = os.getcwd()
FOLDER = r'C:\Users\marcu\source\repos\ConsoleApp1\ConsoleApp1\bin\Debug\frames_out'
POLL_SECONDS = 0.1
DEPTH_CLIP_MM = 4000

# Perception/segmentation params (object)
HEIGHT_THRESH_MM = 5.0
MIN_BLOB_AREA = 300
MAX_BLOB_AREA = 5000
MORPH_KERNEL = (3, 3)
REFIT_PLANE_EVERY_N = 30

# For visualization (world frame)
SHOULDER_XY_WORLD = np.array([-0.07, 0.0], dtype=float)
Z_TABLE_FIXED = 0.05  # meters

def getPositions(show=False):
    os.makedirs(FOLDER, exist_ok=True)
    print("Depth tracker initialized — streaming positions...")

    rvec, tvec = loadExtrinsic(BASEPATH)
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
        print(f"\n--- Frame {frame_idx} ---")

        h, w = depth.shape
        fx, fy, cx, cy = intrinsics_for_shape(w, h)
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)

        if plane_model is None or frames_since_plane >= REFIT_PLANE_EVERY_N:
            plane_model = ransac_floor_plane(depth, fx, fy, cx, cy)
            frames_since_plane = 0
            if plane_model is None:
                print("No plane found.")
                continue
        else:
            frames_since_plane += 1

        X, Y, Z = backproject(depth, fx, fy, cx, cy)
        # height above plane (mm)
        a, b, c, d = plane_model
        H_mm = np.abs(a*X + b*Y + c*Z + d) * 1000.0
        valid = (depth > 0) & (depth < DEPTH_CLIP_MM)

        rgb_img = cv2.imread(rgb_path) if rgb_path else None
        mask_red = redMask(rgb_img) if rgb_img is not None else np.zeros_like(depth, dtype=np.uint8)
        obj_mask = (H_mm > HEIGHT_THRESH_MM) & valid & (mask_red > 0)

        obj_mask_u8 = (obj_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
        obj_mask_u8 = cv2.morphologyEx(obj_mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        obj_mask_u8 = cv2.morphologyEx(obj_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask_u8, connectivity=8)

        idle_ms(10, show)

        for i in range(1, num):
            _, _, _, _, area = stats[i]
            if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
                continue

            mask_i = (labels == i)
            Z_blob = Z[mask_i]; Z_blob = Z_blob[(Z_blob > 0) & np.isfinite(Z_blob)]
            X_blob = X[mask_i]; X_blob = X_blob[np.isfinite(X_blob)]
            Y_blob = Y[mask_i]; Y_blob = Y_blob[np.isfinite(Y_blob)]
            if Z_blob.size == 0 or X_blob.size == 0 or Y_blob.size == 0:
                continue

            x_cam = float(np.median(X_blob)); y_cam = float(np.median(Y_blob)); z_m = float(np.median(Z_blob))
            X_cam = np.array([x_cam, y_cam, z_m], dtype=np.float64).reshape(3, 1)
            X_world = R.T @ (X_cam - tvec.reshape(3, 1))
            Xw = X_world.ravel()

            if show and (rgb_img is not None):
                try:
                    disp = rgb_img.copy()
                    cx_px, cy_px = int(round(centroids[i][0])), int(round(centroids[i][1]))
                    cv2.circle(disp, (cx_px, cy_px), 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                    txt = f"x={Xw[0]:.3f} y={Xw[1]:.3f} z={Xw[2]:.3f} m"
                    cv2.putText(disp, txt, (max(0, cx_px + 8), max(10, cy_px - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                    # Path overlay (world) to help visualize what the sender will do
                    p_world = np.array([Xw[0], Xw[1], Xw[2]], dtype=np.float64)
                    path_world = plan_adaptive(
                        target_xyz=p_world,
                        shoulder_xy=SHOULDER_XY_WORLD,
                        back_offset=0.06, up_offset=0.12,
                        f1=0.25, f2=0.10, shape_p=1.8,
                        n_per_segment=8, z_table=Z_TABLE_FIXED, clearance=0.01
                    )

                    Rcw = cv2.Rodrigues(rvec)[0]
                    tcw = tvec.reshape(3, 1)
                    uv, valid_uv = project_world_points(path_world, K, Rcw, tcw)

                    poly = [(int(round(u)), int(round(v))) for ok, (u, v) in zip(valid_uv, uv) if ok]
                    if len(poly) >= 2:
                        cv2.polylines(disp, [np.array(poly, dtype=np.int32)], False, (0, 255, 255), 3)
                    if poly:
                        cv2.circle(disp, poly[0], 6, (0, 165, 255), -1, cv2.LINE_AA)
                        cv2.putText(disp, "pre-approach", (poly[0][0] + 8, poly[0][1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
                        cv2.circle(disp, poly[-1], 6, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.putText(disp, "target", (poly[-1][0] + 8, poly[-1][1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                    cv2.imshow("Depth tracker - object (red)", disp)

                    prof = render_approach_profile(
                        path_world=path_world,
                        target_world_xyz=p_world,
                        shoulder_xy=SHOULDER_XY_WORLD,
                        z_table=Z_TABLE_FIXED, size=(520, 360)
                    )
                    cv2.imshow("Object approach profile (s vs z)", prof)

                    if cv2.waitKey(30) & 0xFF == 27:
                        cv2.destroyAllWindows()
                        return
                except Exception as e:
                    print("Display error:", e)

            # World->robot frame offsets (same as your previous code)
            x_r = -float(Xw[0]) - 0.03
            y_r =  float(Xw[1])
            z_r =  float(Xw[2]) - 0.04
            yield (x_r, y_r, z_r)

        idle_ms(10, show)
