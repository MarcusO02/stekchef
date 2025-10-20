# overlays.py
import numpy as np

def project_world_points(Pw, K, Rcw, tcw):
    """
    Pw: (N,3) world points
    Rcw, tcw: world->camera extrinsics (3x3, 3x1)
    K: camera intrinsics (3x3)
    Returns:
      uv: (N,2) float pixels (NaN for invalid)
      valid: (N,) bool where depth > 0
    """
    Pw = np.asarray(Pw, dtype=float).reshape(-1, 3)
    Pc = (Rcw @ Pw.T + tcw.reshape(3, 1)).T  # (N,3)
    uv = np.empty((Pw.shape[0], 2), dtype=float)
    valid = Pc[:, 2] > 1e-6
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uv[valid, 0] = fx * (Pc[valid, 0] / Pc[valid, 2]) + cx
    uv[valid, 1] = fy * (Pc[valid, 1] / Pc[valid, 2]) + cy
    uv[~valid] = np.nan
    return uv, valid