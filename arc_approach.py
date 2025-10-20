# arc_approach.py
import math
import numpy as np
import cv2

# -----------------------------
# Arc configuration container
# -----------------------------
class ArcSpec:
    def __init__(self,
                 alpha,           # 0=ren sida, 1=ren topp
                 R,               # m, radie på bågen
                 L,               # m, båglängd (φ = L/R)
                 h_clear,         # m, höjdlyft i början för att kliva över kant
                 N=6,            # antal delsteg på bågen (N+1 punkter)
                 z_table=0.05,    # m, bordets z i world/robot-frame
                 z_margin=0.0,    # m, säkerhetsmarginal över bord
                 ik_limit_buffer_deg=5.0):
        self.alpha = float(alpha)
        self.R = float(R)
        self.L = float(L)
        self.h_clear = float(h_clear)
        self.N = int(N)
        self.z_table = float(z_table)
        self.z_margin = float(z_margin)
        self.ik_limit_buffer_rad = math.radians(ik_limit_buffer_deg)

# -----------------------------
# Helpers
# -----------------------------
def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def _plane_basis(v_a, v_up):
    # Skapar ett ortonormalt baspar (tangent-riktning, upp) i planet
    t = _normalize(v_a)
    u = _normalize(v_up)
    # Se till att de inte är nästan parallella
    if abs(np.dot(t, u)) > 0.98:
        # välj någon godtycklig vinkelrät riktning
        alt = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(alt, u)) > 0.9:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
        t = _normalize(alt - np.dot(alt, u) * u)
    # Bygg ett högerhandsbas
    n = _normalize(np.cross(t, u))    # normal ut ur planet
    t = _normalize(np.cross(u, n))    # gör t exakt ortonormalt
    return t, u, n

# -----------------------------
# Arc generation
# -----------------------------
def generate_arc_points(p_g,           # grasp-position (3,)
                        p_base,        # robotbasens position (3,) – används bara för side-vector
                        n_table,       # bordets normal (3,)
                        spec: ArcSpec):
    """
    Returnerar:
      points_3d: (N+1, 3) från start -> grasp
      meta: dict med start, phi (rad), success flag
    """
    p_g = np.asarray(p_g, dtype=float).reshape(3)
    p_base = np.asarray(p_base, dtype=float).reshape(3)
    n_table = _normalize(np.asarray(n_table, dtype=float).reshape(3))
    # sidriktning: projicera from base->obj på bordets plan
    v_side = p_g - p_base
    v_side = v_side - np.dot(v_side, n_table) * n_table
    v_side = _normalize(v_side) if np.linalg.norm(v_side) > 1e-9 else np.array([1.0, 0.0, 0.0])

    v_top = n_table
    v_a = _normalize((1.0 - spec.alpha) * v_side + spec.alpha * v_top)

    # startpunkt: backa L/2 längs v_a + lyft h_clear
    d_pre = spec.L * 0.5
    p_s = p_g - d_pre * v_a + spec.h_clear * v_top

    # bygg planet och lokal 2D-parametrisering
    t_hat, u_hat, n_hat = _plane_basis(v_a, v_top)  # t_hat ~ v_a, u_hat ~ v_top

    # Representera start och mål i planet (koordinater i bas [t_hat,u_hat,n_hat])
    # Vi lägger bågen i t-u-planet, ignorera n-led (ska bli ~0)
    def to_plane_coords(p):
        # projektion på bas
        return np.array([np.dot(p, t_hat), np.dot(p, u_hat), np.dot(p, n_hat)])

    def from_plane_coords(c):
        return c[0]*t_hat + c[1]*u_hat + c[2]*n_hat

    c_s = to_plane_coords(p_s)
    c_g = to_plane_coords(p_g)

    # Vi vill ha en cirkelbåge med radie R som går från c_s till c_g i t-u-planet
    # Lös center i planet: två cirkelbågar med given R som går genom båda punkter
    ps = c_s[:2]; pg = c_g[:2]
    d = np.linalg.norm(pg - ps)
    if d < 1e-6 or d > 2.0*spec.R:
        return None, {"success": False, "reason": f"invalid chord length d={d:.3f} vs 2R={2*spec.R:.3f}"}

    # mittpunkt på strängen
    mid = 0.5*(ps + pg)
    # höjd från mittpunkt till cirkelcentrum
    h = math.sqrt(max(spec.R**2 - (d*0.5)**2, 0.0))
    # strängens normal i planet
    chord_dir = _normalize(pg - ps)
    # vinkelrät riktning i planet (två möjliga)
    normal_dir = np.array([-chord_dir[1], chord_dir[0]])
    # välj den normal som gör att start->slut är "framåt" längs t_hat (heuristik: välj så att tangent i start pekar ungefär mot v_a)
    # center-kandidater
    c1 = mid + h*normal_dir
    c2 = mid - h*normal_dir

    # Välj center som ger starttangent nära +t_hat (dvs ökar t-komponenten)
    # Tangent i start = rotera radialvektorn 90 grader moturs
    def start_tangent(center):
        radial = ps - center
        tan = np.array([-radial[1], radial[0]])
        return _normalize(tan)
    tan1 = start_tangent(c1)
    tan2 = start_tangent(c2)
    # jämför med +t-riktning i planet => [1,0]
    score1 = np.dot(tan1, np.array([1.0, 0.0]))
    score2 = np.dot(tan2, np.array([1.0, 0.0]))
    center = c1 if score1 >= score2 else c2

    # Vinklar för start och mål relativt center
    def angle_of(v):
        return math.atan2(v[1], v[0])
    th_s = angle_of(ps - center)
    th_g = angle_of(pg - center)

    # Bestäm svepriktning så vi går "framåt" längs t-axeln (samma kriterium som ovan)
    # Vi väljer minsta positiva svep som når th_g från th_s med tangent i start ~ +t.
    def unwrap(a):
        # så vi kan justera kontinuerligt
        return a
    # test två alternativ: öka vinkel (ccw) eller minska (cw)
    def sweep_len_ccw(a0, a1):
        d = a1 - a0
        while d <= 0: d += 2*math.pi
        return d
    def sweep_len_cw(a0, a1):
        d = a0 - a1
        while d <= 0: d += 2*math.pi
        return d

    phi_ccw = sweep_len_ccw(th_s, th_g)
    phi_cw  = sweep_len_cw(th_s, th_g)

    # välj det som ger tangent närmare +t i start (vi har redan valt center så det bör matcha ccw oftast)
    phi = phi_ccw if score1 >= score2 else phi_cw
    # klipp/varna om phi skiljer sig mycket från önskad L/R:
    phi_target = spec.L / spec.R
    # vi kan låta phi bli vad geometrin kräver; rapportera bara
    meta = {"phi": phi, "phi_target": phi_target}

    # Diskretisera bågen
    pts = []
    steps = spec.N
    sgn = 1.0 if phi == phi_ccw else -1.0
    for k in range(steps+1):
        a = th_s + sgn * phi * (k/steps)
        c2d = center + np.array([math.cos(a)*spec.R, math.sin(a)*spec.R])
        c3d = np.array([c2d[0], c2d[1], 0.0])
        p = from_plane_coords(c3d)
        pts.append(p)
    pts = np.vstack(pts)  # (N+1, 3)

    # Z-säkerhet
    zmin = spec.z_table + spec.z_margin
    if np.any(pts[:,2] < zmin - 1e-4):
        return None, {"success": False, "reason": "z below table margin"}

    return pts, {"success": True, **meta, "start": p_s, "end": p_g, "center_plane": center}
    

# -----------------------------
# Orientation along tangent (optional: yaw only)
# -----------------------------
def orientations_along_arc(points):
    """
    Returnerar en lista av lokala framåtriktningar (tangenter) som kan
    användas för att sätta EE-orientering. Här returnerar vi bara tangenterna.
    """
    if points is None or len(points) < 2:
        return None
    tangents = []
    for i in range(len(points)):
        if i == len(points)-1:
            t = points[i] - points[i-1]
        else:
            t = points[i+1] - points[i]
        tangents.append(_normalize(t))
    return np.vstack(tangents)  # (N+1,3)

# -----------------------------
# Projection (world->image) and overlay
# -----------------------------
def project_world_points(Pw, K, Rcw, tcw):
    """
    Pw: (N,3) world
    Rcw, tcw: world->camera (3x3, 3x1)
    Return: list of (u, v) ints, and a mask of valid points (Z>0 and inside image if w,h given)
    """
    Pw = np.asarray(Pw, dtype=float).reshape(-1,3)
    Pc = (Rcw @ Pw.T + tcw.reshape(3,1)).T  # (N,3)
    uv = np.empty((Pw.shape[0], 2), dtype=float)
    valid = Pc[:,2] > 1e-6
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    uv[valid,0] = fx * (Pc[valid,0] / Pc[valid,2]) + cx
    uv[valid,1] = fy * (Pc[valid,1] / Pc[valid,2]) + cy
    uv[~valid] = np.nan
    return uv, valid

def draw_arc_overlay_verbose(img_bgr, uv, valid, *,
                             color_line=(0,255,255),  # yellow line
                             thickness=3,
                             point_radius=5,
                             show_indices=True,
                             start_color=(0,180,255),  # orange
                             mid_color=(255,255,255),  # white
                             end_color=(0,0,255)):     # red
    """
    Draws the full arc more clearly:
      • Thick polyline
      • Color-coded start/middle/end points
      • Optional point indices (0,1,2,...) for debugging
    """
    out = img_bgr.copy()
    pts = []
    for (u,v), ok in zip(uv, valid):
        if not ok or not np.isfinite(u) or not np.isfinite(v):
            pts.append(None)
        else:
            pts.append((int(round(u)), int(round(v))))

    # Draw polyline
    poly = [p for p in pts if p is not None]
    if len(poly) >= 2:
        cv2.polylines(out, [np.array(poly, dtype=np.int32)],
                      isClosed=False, color=color_line, thickness=thickness, lineType=cv2.LINE_AA)

    # Draw start/mid/end points
    if poly:
        cv2.circle(out, poly[0], point_radius+1, start_color, -1, lineType=cv2.LINE_AA)
        mid_idx = len(poly)//2
        cv2.circle(out, poly[mid_idx], point_radius, mid_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, poly[-1], point_radius+1, end_color, -1, lineType=cv2.LINE_AA)

    # Optional index labels
    if show_indices:
        for k, p in enumerate(pts):
            if p is None: continue
            cv2.putText(out, str(k), (p[0]+4, p[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(out, str(k), (p[0]+4, p[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return out

def render_profile_view(points_world,
                        p_ref,
                        v_a,           # angreppsriktning (enhetsvektor)
                        n_table,       # bordets normal (enhetsvektor, upp)
                        z_table,       # bordshöjd (m) i world-frame
                        pan_rim_h=None,# valfri: pannans kant-höjd (m) över bordet
                        width_px=520,
                        height_px=360,
                        s_pad_m=0.03,  # marginal i s-led (m)
                        z_pad_m=0.02   # marginal i z-led (m)
                        ):

    Pw = np.asarray(points_world, float).reshape(-1, 3)
    p_ref = np.asarray(p_ref, float).reshape(3)
    v_a = v_a / (np.linalg.norm(v_a) + 1e-12)
    n_table = n_table / (np.linalg.norm(n_table) + 1e-12)

    # Basvektorer för profilen
    t_hat = v_a                  # horisontell i profil (fram/bak längs angrepp)
    u_hat = n_table              # vertikal i profil (uppåt)
    # s-koordinat relativt p_ref, z absolut (dot mot u_hat):
    s_vals = (Pw - p_ref) @ t_hat
    z_vals = Pw @ u_hat

    # Intervall + padding
    s_min, s_max = s_vals.min(), s_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()

    s_min -= s_pad_m; s_max += s_pad_m
    z_min = min(z_min, z_table) - z_pad_m
    z_max = max(z_max, z_table + (pan_rim_h or 0.0)) + z_pad_m

    # Undvik degenererat intervall
    if abs(s_max - s_min) < 1e-6:
        s_max = s_min + 0.05
    if abs(z_max - z_min) < 1e-6:
        z_max = z_min + 0.05

    # Skala till pixlar (vänster->höger är s_min->s_max, ner->upp är z_min->z_max)
    def to_px(s, z):
        u = int(round((s - s_min) / (s_max - s_min) * (width_px - 1)))
        v = int(round((1.0 - (z - z_min) / (z_max - z_min)) * (height_px - 1)))
        return u, v

    img = np.full((height_px, width_px, 3), 255, np.uint8)

    # Rutnät / axlar
    # z= bordshöjd
    u0, v_table = to_px(s_min, z_table)
    u1, _       = to_px(s_max, z_table)
    cv2.line(img, (u0, v_table), (u1, v_table), (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(img, f"table z={z_table:.3f} m", (6, max(14, v_table-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1, cv2.LINE_AA)

    # ev. pannkantslinje
    if pan_rim_h is not None:
        z_rim = z_table + pan_rim_h
        u0, v_rim = to_px(s_min, z_rim)
        u1, _     = to_px(s_max, z_rim)
        cv2.line(img, (u0, v_rim), (u1, v_rim), (160,160,220), 1, cv2.LINE_AA)
        cv2.putText(img, f"rim z={z_rim:.3f} m", (6, max(28, v_rim-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,200), 1, cv2.LINE_AA)

    # Polyline för bågen
    poly = [to_px(s, z) for s, z in zip(s_vals, z_vals)]
    cv2.polylines(img, [np.array(poly, np.int32)], False, (0,180,255), 2, cv2.LINE_AA)

    # Start / mitt / slut-markörer
    if poly:
        cv2.circle(img, poly[0], 5, (0,180,255), -1, cv2.LINE_AA)     # start
        cv2.circle(img, poly[len(poly)//2], 5, (0,0,0), -1, cv2.LINE_AA)  # mitt
        cv2.circle(img, poly[-1], 5, (0,0,255), -1, cv2.LINE_AA)      # slut

    # Minsta höjdmarginal mot bordet
    min_clear = float((z_vals - z_table).min())
    cv2.putText(img, f"min clearance: {min_clear*1000:.0f} mm",
                (6, height_px-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,120,0), 1, cv2.LINE_AA)

    # s- och z-etiketter
    cv2.putText(img, "s (along approach)", (width_px//2 - 80, height_px-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1, cv2.LINE_AA)
    cv2.putText(img, "z (height)", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1, cv2.LINE_AA)

    return img


