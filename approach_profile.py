# profile.py
import numpy as np
import cv2
from approach_path import unit_xy_from_shoulder

def render_approach_profile(
    path_world: np.ndarray,
    target_world_xyz: np.ndarray,
    shoulder_xy: np.ndarray,
    z_table: float | None = None,
    size: tuple[int, int] = (520, 360)
) -> np.ndarray:
    """
    Returns a BGR image showing s (along shoulder->target) vs z (height).
    s = 0 at target, negative behind target.
    """
    W, H = size
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (30, 30, 30)

    tx, ty, tz = map(float, target_world_xyz[:3])
    dir_xy = unit_xy_from_shoulder(target_world_xyz, np.asarray(shoulder_xy))

    # Project each waypoint to (s, z)
    s_vals, z_vals = [], []
    for p in path_world:
        dx, dy = float(p[0] - tx), float(p[1] - ty)
        s = dx * dir_xy[0] + dy * dir_xy[1]
        s_vals.append(s)
        z_vals.append(float(p[2]))
    s_vals = np.array(s_vals)
    z_vals = np.array(z_vals)

    s_min, s_max = float(np.min(s_vals)), float(np.max(s_vals))
    if s_max <= s_min + 1e-9:
        s_max = s_min + 1e-3
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
    if z_table is not None:
        z_min = min(z_min, z_table)
        z_max = max(z_max, z_table)
    if z_max <= z_min + 1e-9:
        z_max = z_min + 1e-3

    L, R, T, B = 60, 20, 20, 40
    plot_w = W - L - R
    plot_h = H - T - B

    def to_px(s, z):
        u = (s - s_min) / (s_max - s_min)
        v = (z - z_min) / (z_max - z_min)
        x = int(round(L + u * plot_w))
        y = int(round(T + (1 - v) * plot_h))
        return x, y

    cv2.rectangle(img, (L, T), (L + plot_w, T + plot_h), (80, 80, 80), 1)
    x0, _ = to_px(0.0, z_min)
    cv2.line(img, (x0, T), (x0, T + plot_h), (70, 120, 200), 1)
    if z_table is not None:
        _, ytab = to_px(s_min, z_table)
        cv2.line(img, (L, ytab), (L + plot_w, ytab), (100, 100, 160), 1)

    poly = [to_px(s, z) for s, z in zip(s_vals, z_vals)]
    if len(poly) >= 2:
        cv2.polylines(img, [np.array(poly, np.int32)], False, (0, 255, 255), 2)
    if poly:
        cv2.circle(img, poly[0], 5, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(img, poly[-1], 5, (0, 255, 0), -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "s (m)", (L + plot_w - 50, H - 8), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "z (m)", (8, T + 12), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"s_min={s_min:.3f}", (L, 16), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(img, f"z[{z_min:.2f},{z_max:.2f}]", (L + 140, 16), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    if z_table is not None:
        cv2.putText(img, f"table z={z_table:.3f}", (L, H - 12), font, 0.45, (140, 140, 200), 1, cv2.LINE_AA)

    return img