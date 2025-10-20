# approach_path.py
import numpy as np

def unit_xy_from_shoulder(target_xyz: np.ndarray, shoulder_xy: np.ndarray) -> np.ndarray:
    """
    Unit vector in XY from the shoulder to the target, in the SAME frame as target_xyz.
    """
    tx, ty = float(target_xyz[0]), float(target_xyz[1])
    sx, sy = float(shoulder_xy[0]), float(shoulder_xy[1])
    v = np.array([tx - sx, ty - sy], dtype=float)
    n = np.linalg.norm(v) + 1e-9
    return v / n

def plan_straight(
    target_xyz: np.ndarray,
    shoulder_xy: np.ndarray,
    *,
    back_offset: float = 0.08,
    up_offset: float = 0.05,
    n: int = 20
) -> np.ndarray:
    """
    Simple: start behind+above target, then straight line to target, in the shoulder→target plane.
    """
    tx, ty, tz = map(float, target_xyz[:3])
    u_xy = unit_xy_from_shoulder(target_xyz, shoulder_xy)
    e1 = np.array([u_xy[0], u_xy[1], 0.0], float)
    start = np.array([tx, ty, tz]) - back_offset * e1 + np.array([0.0, 0.0, up_offset])
    t = np.linspace(0.0, 1.0, num=max(2, int(n)))
    pts = (1 - t)[:, None] * start + t[:, None] * np.array([tx, ty, tz])
    return pts.astype(float)

def plan_adaptive(
    target_xyz: np.ndarray,
    shoulder_xy: np.ndarray,
    *,
    back_offset: float = 0.15,
    up_offset: float = 0.06,
    f1: float = 0.60,
    f2: float = 0.25,
    shape_p: float = 2.1,
    n_per_segment: int = 8,
    z_table: float | None = None,
    clearance: float = 0.010
) -> np.ndarray:
    """
    4-key-point, piecewise-linear path that flattens toward the target and never dips below
    max(target_z+clearance, table+clearance). Returns interpolated Nx3 waypoints.
    """
    tx, ty, tz = map(float, target_xyz[:3])

    # Along-line basis (XY toward target) and path length S
    u_xy = unit_xy_from_shoulder(target_xyz, shoulder_xy)
    e1 = np.array([u_xy[0], u_xy[1], 0.0], float)
    S  = max(1e-4, float(back_offset))
    up = max(0.0, float(up_offset))
    f1 = float(np.clip(f1, 0.0, 1.0))
    f2 = float(np.clip(f2, 0.0, f1))
    p  = max(1.0, float(shape_p))

    # Height floor
    z_floor = tz + float(clearance)
    if z_table is not None:
        z_floor = max(z_floor, float(z_table) + float(clearance))

    def z_of_s(s: float) -> float:
        return max(tz + up * (s / S) ** p, z_floor)

    # Distances from start (behind) to target
    s0, s1, s2, s3 = S, S * f1, S * f2, 0.0
    z0, z1, z2, z3 = z_of_s(s0), z_of_s(s1), z_of_s(s2), tz

    # Enforce flattening: |m01| >= |m12| >= |m23|
    def slope(sa, za, sb, zb):
        return abs((zb - za) / (sb - sa)) if sb != sa else 0.0
    m01 = slope(s0, z0, s1, z1)
    m12 = slope(s1, z1, s2, z2)
    m23 = slope(s2, z2, s3, z3)

    if m23 > m12 + 1e-6:
        z2 = max(z3 + m12 * s2, z_floor)
        m12 = slope(s1, z1, s2, z2)
        if m12 > m01 + 1e-6:
            z1 = max(z2 - m01 * (s2 - s1), z_floor)

    # Key points
    P0 = np.array([tx, ty, tz]) - s0 * e1; P0[2] = z0
    P1 = np.array([tx, ty, tz]) - s1 * e1; P1[2] = z1
    P2 = np.array([tx, ty, tz]) - s2 * e1; P2[2] = z2
    P3 = np.array([tx, ty, tz]) - s3 * e1; P3[2] = z3

    # Interpolate segments
    key_pts = [P0, P1, P2, P3]
    path = []
    for A, B in zip(key_pts[:-1], key_pts[1:]):
        for j in range(max(1, int(n_per_segment))):
            t = (j + 1) / n_per_segment
            path.append((1 - t) * A + t * B)
    return np.vstack(path)