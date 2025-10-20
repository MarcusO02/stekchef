#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hubert_ik.py
-------------
Compute servo pulse values (µs) for Hubert robot arm
given a target (x, y, z) position in meters.

This module is designed to be imported by send_to_robot.py
or other scripts that provide live coordinates (e.g. depth tracker).

Example:
    from hubert_ik import compute_pulses_from_xyz, build_chain
    chain = build_chain()
    pulses, ordered = compute_pulses_from_xyz(0.2, 0.2, 0.2, chain)
    print(pulses, ordered)
"""

import math
import numpy as np
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET
from ikpy.chain import Chain

# -------------------------------------------------------------
# === Configuration ===
# -------------------------------------------------------------

# Path to your robot description
URDF_PATH = "hubert_stekchef.urdf"

# Names of joints to use for IK
ACTUATED_JOINTS = {"base_yaw", "shoulder_pitch", "elbow_pitch"}

# Servo pulse limits (µs)
SERVO_US = {
    "elbow_pitch":    (1160, 2110),
    "base_yaw":       (560,  2310),
    "shoulder_pitch": (1550, 2180),
}

# Whether the servo direction is inverted
SERVO_INVERT = {
    "elbow_pitch":    False,
    "base_yaw":       True,
    "shoulder_pitch": True,
}

# Fallback joint limits (radians) in case URDF doesn’t define them
FALLBACK_LIMITS = {
    "base_yaw":       (-1.5708, 1.5708),   # ~±90°
    "shoulder_pitch": (-0.5410, 0.5236),   # ~-31°..+30°
    "elbow_pitch":    (0.0,     1.5708),   # 0..90°
}

# Order to send data to Arduino
SERIAL_ORDER = ["base_yaw", "shoulder_pitch", "elbow_pitch"]

# -------------------------------------------------------------
# === URDF Utilities ===
# -------------------------------------------------------------

def sanitize_urdf_for_ikpy(urdf_text: str) -> str:
    """
    Convert continuous joints to revolute and add limits if missing.
    Makes URDF compatible with IKPy.
    """
    root = ET.fromstring(urdf_text)
    for joint in root.findall("joint"):
        jtype = (joint.get("type") or "").strip().lower()
        if jtype == "continuous":
            joint.set("type", "revolute")
            if joint.find("axis") is None:
                axis = ET.SubElement(joint, "axis")
                axis.set("xyz", "0 0 1")
            if joint.find("limit") is None:
                lim = ET.SubElement(joint, "limit")
                lim.set("lower", f"{-math.pi}")
                lim.set("upper", f"{ math.pi}")
                lim.set("effort", "50")
                lim.set("velocity", "5.0")
    return ET.tostring(root, encoding="unicode")

def build_chain():
    """
    Build and return the IKPy Chain object from the URDF.
    """
    urdf_path = Path(URDF_PATH)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    urdf_text = urdf_path.read_text(encoding="utf-8")

    # Clean and write to temp file
    safe_urdf = sanitize_urdf_for_ikpy(urdf_text)
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
    tmp.write(safe_urdf.encode("utf-8"))
    tmp.close()

    chain = Chain.from_urdf_file(tmp.name, base_elements=["base_fixed"])

    # Trim chain to include links up to end effector
    names = [getattr(l, "name", "") for l in chain.links]
    if "end_effector" in names:
        ee_idx = names.index("end_effector")
        chain.links = chain.links[:ee_idx + 1]

    # Mask for actuated joints
    mask = [(getattr(l, "name", "") in ACTUATED_JOINTS) for l in chain.links]
    chain.active_links_mask = mask
    return chain

# -------------------------------------------------------------
# === IK + Mapping Logic ===
# -------------------------------------------------------------

def get_joint_limits(link):
    lb = getattr(link, "lower_limit", None)
    ub = getattr(link, "upper_limit", None)
    if lb is None or ub is None:
        return FALLBACK_LIMITS.get(getattr(link, "name", ""), (-math.pi, math.pi))
    return float(lb), float(ub)

def angle_to_us(angle, lower, upper, us_min, us_max, invert=False):
    """
    Map a joint angle (radians) to a servo pulse width (µs).
    """
    angle = np.clip(angle, lower, upper)
    if invert:
        us_min, us_max = us_max, us_min
    t = (angle - lower) / (upper - lower)
    return int(round(us_min + t * (us_max - us_min)))

def compute_pulses_from_xyz(x, y, z, chain=None):
    """
    Compute servo pulses (µs) for a given Cartesian position (x, y, z).
    Returns:
        (pulses_dict, ordered_pulse_list)
    """
    if chain is None:
        chain = build_chain()

    # Inverse kinematics
    q = chain.inverse_kinematics(target_position=[x, y, z])

    pulses = {}
    for link, angle in zip(chain.links, q):
        name = getattr(link, "name", "")
        if name in ACTUATED_JOINTS:
            lb, ub = get_joint_limits(link)
            us_min, us_max = SERVO_US[name]
            invert = SERVO_INVERT[name]
            pulses[name] = angle_to_us(angle, lb, ub, us_min, us_max, invert)

    ordered = [pulses[n] for n in SERIAL_ORDER]
    return pulses, ordered
