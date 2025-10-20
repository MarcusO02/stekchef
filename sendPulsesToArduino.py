# send_pulses.py
import sys, time, random, serial, numpy as np
import cv2
from enum import Enum, auto

from inverseKinematics import compute_pulses_from_xyz, build_chain
from getPositions import getPositions          # red object stream (pick)
from getPanPositions import getPanPositions    # green/black pan stream (place)
from approach_path import plan_adaptive        # shoulder-aligned adaptive path

sys.stdout.reconfigure(line_buffering=True)

# ---------------- Robot-frame config ----------------
SHOULDER_XY_ROBOT = np.array([-0.1079, 0.0], dtype=float)
SHOULDER_XY_ROBOT2 = np.array([0.05, 0.0], dtype=float)

# Home pose
HOME_US = (1405, 1914, 2110)

# Serial/handshake
READY_TIMEOUT_S = 3.0

# Staging point near the pan
PAN_STAGING_XYZ = np.array([0.00, 0.29, 0.25], dtype=float)
PAN_STAGING_XYZ2 = np.array([-0.19, 0.19, 0.175], dtype=float)

# Store initial pickup position
INITIAL_PICKUP_POS = None  # Will be set when first detecting the object


# Direct place height above detected pan center
PAN_Z_OFFSET = 0.15

# Store initial pickup position
INITIAL_PICKUP_POS = None  # Will be set when first detecting the object

# Wrist presets (µs)
WRIST_ALIGN_PICK_US   = 100
WRIST_ALIGN_PLACE_US  = 100
WRIST_ROTATE_US       = 2300   # rotate during "toss/flip" part; tune

PUSH_DXY = np.array([-0.03, 0.03], dtype=float)  # +x = forward, +y = left (tune)
PUSH_DEPTH = 0.15  # ~2 cm; tune conservatively
PUSH_LIFT_AFTER = 0.15
PUSH_TARGET_ABS_XY = None



# ---------------- Serial helpers ----------------
def send_to_arduino(ordered_pulses, ser):
    line = ",".join(str(v) for v in ordered_pulses) + "\n"
    ser.write(line.encode("ascii"))

def send_gripper(state: str, ser):  # "OPEN" or "CLOSE"
    ser.write(f"GRIP,{state}\n".encode("ascii"))

def send_wrist_us(us, ser):
    ser.write(f"WRIST,{int(us)}\n".encode("ascii"))

def wait_for_ready(ser, timeout_s=READY_TIMEOUT_S, pump_gui=True):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode(errors="ignore").strip()
        if pump_gui:
            try: cv2.waitKey(1)
            except Exception: pass
        if not line:
            continue
        if line.startswith("READY"):
            return True
        if line.startswith("OK") or line.startswith("ERR"):
            print(line)
    return False

# ---------------- Helpers ----------------
# Put near top of send_pulses.py
TCP_FORWARD_OFFSET = -0.020   # 20 mm forward of your modeled TCP (tune)
TCP_LATERAL_OFFSET = 0.030   # if you know it’s a bit left/right, add here too
TCP_Z_OFFSET       = 0.000   # vertical bias if needed (usually 0)

def yaw_rotated_tcp_offset(target_xyz, shoulder_xy):
    # base/shoulder->target yaw in XY
    v = target_xyz[:2] - shoulder_xy
    a = np.arctan2(v[1], v[0])  # yaw
    ca, sa = np.cos(a), np.sin(a)
    # tool offsets expressed in the approach frame (forward=x', left=y')
    dxp, dyp, dz = TCP_FORWARD_OFFSET, TCP_LATERAL_OFFSET, TCP_Z_OFFSET
    # rotate into robot/world XY (z unchanged)
    dx =  ca*dxp - sa*dyp
    dy =  sa*dxp + ca*dyp
    return np.array([dx, dy, dz], dtype=float)

def pulses_for_xyz(x, y, z, chain):
    _, ordered = compute_pulses_from_xyz(float(x), float(y), float(z), chain)
    return ordered

def goto_xyz(pt_xyz, chain, ser=None, test=False, label=""):
    pulses = pulses_for_xyz(pt_xyz[0], pt_xyz[1], pt_xyz[2], chain)
    if ser is not None and not test:
        send_to_arduino(pulses, ser)
        ok = wait_for_ready(ser)
        if not ok:
            print(f"{label} READY timeout; continuing")
    else:
        print(f"{label} pulses: {pulses}")
        time.sleep(0.2)

def get_mock_positions():
    while True:
        yield (random.uniform(0.15, 0.30),
               random.uniform(0.10, 0.25),
               random.uniform(0.15, 0.25))

def wait_for_next(gen, timeout_s=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            return next(gen)
        except StopIteration:
            break
        try: cv2.waitKey(1)
        except: pass
    return None

# ---------------- FSM ----------------
class Phase(Enum):
    # First cycle (pick → place)
    INIT_HOME = auto()
    WAIT_TARGET = auto()
    MOVE_TO_PRE = auto()
    ALIGN_OPEN = auto()
    APPROACH = auto()
    GRIP_CLOSE = auto()
    RETREAT = auto()                     
    FIND_PAN = auto()
    MOVE_TO_PAN_STAGING = auto()
    PLACE_DIRECT = auto()
    LOWER_IN_PAN = auto()
    GRIP_OPEN = auto()
    ARC_TOWARD_SHOULDER = auto()
    RETURN_TO_PAN_STAGING = auto()
    MOVE_HOME = auto()

    # Second cycle (re-pick in pan using arc → flip → drop back → home)
    PAUSE_AT_HOME = auto()
    FIND_OBJECT_AGAIN = auto()
    MOVE_TO_PAN_STAGING2 = auto()
    MOVE_TO_PRE2 = auto()
    ALIGN_OPEN2 = auto()
    APPROACH2 = auto()
    GRIP_CLOSE2 = auto()
    LIFT_FROM_PAN = auto()
    WRIST_ROTATE = auto()
    REPLACE_IN_PAN = auto()
    OPEN_AGAIN = auto()
    MOVE_TO_PRE3 = auto()
    MOVE_HOME_FINAL = auto()
    
    # Return to original position cycle
    RETURN_TO_PAN = auto()
    MOVE_TO_PAN_STAGING3 = auto()
    APPROACH_PAN_PICKUP_START = auto()
    APPROACH_PAN_PICKUP = auto()
    GRIP_CLOSE_PAN = auto()
    LIFT_FROM_PAN_RETURN = auto()
    RETURN_TO_PAN_STAGING_ORG = auto()
    MOVE_TO_ORIGINAL_STAGING = auto()
    APPROACH_ORIGINAL = auto()
    RELEASE_AT_ORIGINAL = auto()
    RETREAT_FINAL = auto()
    DONE = auto()

def main(test=False, port="COM3"):
    # Serial
    ser = None
    if not test:
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"Connected to Arduino on {port}")
        except Exception as e:
            print(f"Could not connect to Arduino: {e}")
            ser = None

    chain = build_chain()

    show_flag = "--show" in sys.argv
    # Perception streams:
    positions_obj = get_mock_positions() if test else getPositions(show=show_flag)
    positions_pan = get_mock_positions() if test else getPanPositions(show=show_flag)

    phase = Phase.INIT_HOME
    target_xyz = None
    pan_xyz = None
    last_pan_xyz = None  # remember pan height for second arc plan
    path = None
    path2 = None
    path_shoulder = None

    try:
        while True:
            # -------- First cycle --------
            if phase == Phase.INIT_HOME:
                print("[INIT_HOME] Moving to home pose")
                if ser is not None and not test:
                    send_to_arduino(HOME_US, ser)
                    wait_for_ready(ser)
                phase = Phase.WAIT_TARGET

            elif phase == Phase.WAIT_TARGET:
                print("[WAIT_TARGET] Waiting for object...")
                got = wait_for_next(positions_obj, timeout_s=5.0)
                if not got:
                    print("No object found. Still waiting...")
                    continue
                x, y, z = got
                target_xyz = np.array([x, y, z], dtype=float)
                # Store initial pickup position
                global INITIAL_PICKUP_POS
                INITIAL_PICKUP_POS = target_xyz.copy()
                print(f"  Object: {x:.3f}, {y:.3f}, {z:.3f}")
                target_xyz_tuned = target_xyz + yaw_rotated_tcp_offset(target_xyz, SHOULDER_XY_ROBOT2)

                path = plan_adaptive(
                    target_xyz=target_xyz_tuned,
                    shoulder_xy=SHOULDER_XY_ROBOT2,
                    back_offset=0.18,
                    up_offset=0.05,
                    f1=0.60, f2=0.3,
                    shape_p=2.1,
                    n_per_segment=8,
                    z_table=0.02,
                    clearance=0.010
                )
                phase = Phase.MOVE_TO_PRE

            elif phase == Phase.MOVE_TO_PRE:
                print("[MOVE_TO_PRE] Going to pre-approach (object)")
                goto_xyz(path[0], chain, ser, test, label="Pre (object)")
                phase = Phase.ALIGN_OPEN

            elif phase == Phase.ALIGN_OPEN:
                print("[ALIGN_OPEN] Open gripper and align wrist for pick")
                if ser is not None and not test:
                    send_gripper("OPEN", ser)
                    send_wrist_us(WRIST_ALIGN_PICK_US, ser)
                    time.sleep(2)
                else:
                    print(f"Gripper: OPEN (sim), Wrist: {WRIST_ALIGN_PICK_US}us (sim)")
                phase = Phase.APPROACH

            elif phase == Phase.APPROACH:
                print("[APPROACH] Streaming along path to object")
                for (px, py, pz) in path:
                    goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Waypoint")
                phase = Phase.GRIP_CLOSE

            elif phase == Phase.GRIP_CLOSE:
                print("[GRIP_CLOSE] Close gripper at object")
                if ser is not None and not test:
                    send_gripper("CLOSE", ser)
                    time.sleep(0.25)
                else:
                    print("Gripper: CLOSE (sim)")
                phase = Phase.RETREAT

            elif phase == Phase.RETREAT:
                print("[RETREAT] Returning along path to pre-approach")
                for (px, py, pz) in path[::-1]:
                    goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Retreat")
                phase = Phase.FIND_PAN

            elif phase == Phase.FIND_PAN:
                print("[FIND_PAN] Waiting for pan location...")
                got = wait_for_next(positions_pan, timeout_s=5.0)
                if not got:
                    print("Pan not found → going HOME then DONE.")
                    phase = Phase.MOVE_HOME
                    continue
                px, py, pz = got
                pan_xyz = np.array([px, py, pz], dtype=float)
                last_pan_xyz = pan_xyz.copy()
                print(f"  Pan: {px:.3f}, {py:.3f}, {pz:.3f}")
                phase = Phase.MOVE_TO_PAN_STAGING

            elif phase == Phase.MOVE_TO_PAN_STAGING:
                print("[MOVE_TO_PAN_STAGING] Moving to staging near pan")
                goto_xyz(PAN_STAGING_XYZ, chain, ser, test, label="Pan staging")
                phase = Phase.PLACE_DIRECT

            elif phase == Phase.PLACE_DIRECT:
                print("[PLACE_DIRECT] Direct move to pan (Z offset)")
                place_xyz = np.array([pan_xyz[0], pan_xyz[1], pan_xyz[2] + PAN_Z_OFFSET], dtype=float)
                goto_xyz(place_xyz, chain, ser, test, label="Place target (+Z)")
                phase = Phase.LOWER_IN_PAN

            elif phase == Phase.LOWER_IN_PAN:
                print("[LOWER_IN_PAN] Lowering into pan")
                lower = np.array([last_pan_xyz[0], last_pan_xyz[1], last_pan_xyz[2] + PAN_Z_OFFSET - 0.2], dtype=float)
                goto_xyz(lower, chain, ser, test, label="Lower into pan")
                time.sleep(1)
                phase = Phase.GRIP_OPEN

            elif phase == Phase.GRIP_OPEN:
                print("[GRIP_OPEN] Open to release")
                if ser is not None and not test:
                    send_gripper("OPEN", ser)
                    time.sleep(0.2)
                else:
                    print("Gripper: OPEN (sim)")
                phase = Phase.ARC_TOWARD_SHOULDER

            elif phase == Phase.ARC_TOWARD_SHOULDER:
                print("[ARC_TOWARD_SHOULDER] Arcing away from pan toward shoulder")
                if last_pan_xyz is None:
                    print("  No pan height known; skipping arc toward shoulder.")
                    phase = Phase.RETURN_TO_PAN_STAGING
                    continue

                # Start at the lowered position used during release
                start_pt = np.array([
                    float(last_pan_xyz[0]),
                    float(last_pan_xyz[1]),
                    float(last_pan_xyz[2] + PAN_Z_OFFSET - 0.2)
                ], dtype=float)

                # Prevent dipping below the pan: keep table just under pan height
                z_table_for_arc = float(last_pan_xyz[2] - 0.065)

                # Apply TCP bias so approach vector aligns with tool yaw relative to shoulder
                tcp_bias = yaw_rotated_tcp_offset(start_pt, SHOULDER_XY_ROBOT2)
                start_pt_biased = start_pt + tcp_bias

                # Plan an approach path that would go INTO start_pt from the shoulder direction
                # We'll then traverse it in reverse to move AWAY from the pan toward the shoulder.
                path_shoulder = plan_adaptive(
                    target_xyz=start_pt_biased,
                    shoulder_xy=SHOULDER_XY_ROBOT2,
                    back_offset=0.08,
                    up_offset=0.10,
                    f1=0.35, f2=0.30,
                    shape_p=1.8,
                    n_per_segment=8,
                    z_table=z_table_for_arc,
                    clearance=0.00
                )

                if path_shoulder is None or len(path_shoulder) < 2:
                    print("  Could not compute arc path; proceeding to staging.")
                    phase = Phase.RETURN_TO_PAN_STAGING
                    continue

                # Follow arc away from the pan toward the shoulder direction
                for (px, py, pz) in path_shoulder[::-1]:
                    goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Arc-away to shoulder")
                phase = Phase.RETURN_TO_PAN_STAGING

            elif phase == Phase.RETURN_TO_PAN_STAGING:
                print("[RETURN_TO_PAN_STAGING] Back to staging")
                goto_xyz(PAN_STAGING_XYZ2, chain, ser, test, label="Back to staging")
                phase = Phase.MOVE_HOME


            elif phase == Phase.MOVE_HOME:
                print("[MOVE_HOME] Going to home pose")
                if ser is not None and not test:
                    send_to_arduino(HOME_US, ser)
                    wait_for_ready(ser)
                else:
                    print(f"Home pulses: {HOME_US}")
                    time.sleep(0.2)
                phase = Phase.PAUSE_AT_HOME

            # -------- Second cycle (re-pick with ARC from the pan) --------
            elif phase == Phase.PAUSE_AT_HOME:
                print("[PAUSE_AT_HOME] Waiting 5 seconds before re-pick")
                time.sleep(5.0)
                phase = Phase.FIND_OBJECT_AGAIN

            elif phase == Phase.FIND_OBJECT_AGAIN:
                print("[FIND_OBJECT_AGAIN] Locate red object again (now in pan)")
                got = wait_for_next(positions_obj, timeout_s=5.0)
                if not got:
                    print("Object not found in pan. Ending.")
                    phase = Phase.MOVE_HOME_FINAL
                    continue
                x2, y2, z2 = got
                target2 = np.array([x2, y2, z2], dtype=float)
                print(f"  Object-again: {x2:.3f}, {y2:.3f}, {z2:.3f}")

                # ARC path again; use pan height as z_table so it never dips below the pan
                z_table_for_arc = float(last_pan_xyz[2]-0.065) if last_pan_xyz is not None else 0.05
                tcp_bias = yaw_rotated_tcp_offset(target2, SHOULDER_XY_ROBOT)
                target2_biased = target2 + tcp_bias

                path2 = plan_adaptive(
                    target_xyz=target2_biased,
                    shoulder_xy=SHOULDER_XY_ROBOT2,
                    back_offset=0.08,
                    up_offset=0.12,
                    f1=0.35, f2=0.3,
                    shape_p=1.8,
                    n_per_segment=8,
                    z_table=z_table_for_arc,
                    clearance=0.00
                )
                phase = Phase.MOVE_TO_PAN_STAGING2
            
            elif phase == Phase.MOVE_TO_PAN_STAGING2:
                print("[MOVE_TO_PAN_STAGING] Moving to staging near pan")
                goto_xyz(PAN_STAGING_XYZ2, chain, ser, test, label="Pan staging")
                time.sleep(2)
                phase = Phase.MOVE_TO_PRE2

            elif phase == Phase.MOVE_TO_PRE2:
                print("[MOVE_TO_PRE2] Go to pre-approach (object-in-pan)")
                goto_xyz(path2[0], chain, ser, test, label="Pre (again)")
                phase = Phase.ALIGN_OPEN2

            elif phase == Phase.ALIGN_OPEN2:
                print("[ALIGN_OPEN2] Open gripper + align wrist for 2nd pick")
                if ser is not None and not test:
                    send_gripper("OPEN", ser)
                    send_wrist_us(WRIST_ALIGN_PICK_US, ser)
                    time.sleep(0.2)
                else:
                    print(f"Gripper: OPEN (sim), Wrist: {WRIST_ALIGN_PICK_US}us (sim)")
                phase = Phase.APPROACH2

            elif phase == Phase.APPROACH2:
                print("[APPROACH2] Arc approach to object in pan")
                for (px, py, pz) in path2:
                    goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Waypoint2")
                phase = Phase.GRIP_CLOSE2

            elif phase == Phase.GRIP_CLOSE2:
                print("[GRIP_CLOSE2] Close gripper (in pan)")
                if ser is not None and not test:
                    send_gripper("CLOSE", ser)
                    time.sleep(0.25)
                else:
                    print("Gripper: CLOSE (sim)")
                phase = Phase.LIFT_FROM_PAN

            elif phase == Phase.LIFT_FROM_PAN:
                print("[LIFT_FROM_PAN] Short lift 4 cm")
                endx, endy, endz = path2[-1]
                lift2 = np.array([endx, endy, endz + 0.15], dtype=float)
                goto_xyz(lift2, chain, ser, test, label="Lift2")
                phase = Phase.WRIST_ROTATE

            elif phase == Phase.WRIST_ROTATE:
                print("[WRIST_ROTATE] Rotate wrist to new orientation")
                time.sleep(1)  # pause before flip
                if ser is not None and not test:
                    send_wrist_us(WRIST_ROTATE_US, ser)
                    time.sleep(5)
                else:
                    print(f"Wrist: {WRIST_ROTATE_US}us (sim)")
                phase = Phase.REPLACE_IN_PAN
            

            elif phase == Phase.REPLACE_IN_PAN:
                print("[REPLACE_IN_PAN] Lower straight down from current post-rotation position")
                cur_x = endx
                cur_y = endy

                if 'target2' in locals() and target2 is not None:
                    # place at the same Z where the object was picked in the pan
                    desired_z = float(target2[2])
                elif last_pan_xyz is not None:
                    # place a little above the pan center height
                    desired_z = float(last_pan_xyz[2] + PAN_Z_OFFSET - 0.02)
                else:
                    # fallback: a small drop relative to the arc end
                    desired_z = float(endz - 0.06)

                lower = np.array([cur_x, cur_y, desired_z], dtype=float)
                goto_xyz(lower, chain, ser, test, label="Lower straight down")
                time.sleep(1)
                phase = Phase.OPEN_AGAIN

            elif phase == Phase.OPEN_AGAIN:
                print("[OPEN_AGAIN] Open gripper to drop again")
                if ser is not None and not test:
                    send_gripper("OPEN", ser)
                    time.sleep(0.2)
                else:
                    print("Gripper: OPEN (sim)")
                phase = Phase.MOVE_TO_PRE3
            
            elif phase == Phase.MOVE_TO_PRE3:
                print("[MOVE_TO_PRE3] Go to pre-approach")
                # Follow the entire second arc in reverse (walk back along path2)
                if path2 is not None:
                    for (px, py, pz) in path2[::-1]:
                        goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Pre (again reverse)")
                else:
                    # Fallback: if no path2 available, go to its first waypoint if present
                    print("Warning: path2 is None, falling back to safest known point")
                    time.sleep(0.1)
                phase = Phase.MOVE_HOME_FINAL

            elif phase == Phase.MOVE_HOME_FINAL:
                print("[MOVE_HOME_FINAL] Return home before return cycle")
                if ser is not None and not test:
                    send_to_arduino(HOME_US, ser)
                    wait_for_ready(ser)
                else:
                    print(f"Home pulses: {HOME_US}")
                    time.sleep(2)
                phase = Phase.RETURN_TO_PAN

            elif phase == Phase.RETURN_TO_PAN:
                print("[RETURN_TO_PAN] Starting return cycle - locating object in pan")
                got = wait_for_next(positions_obj, timeout_s=5.0)
                if not got:
                    print("Object not found in pan. Ending.")
                    phase = Phase.MOVE_HOME_FINAL
                    continue
                px, py, pz = got
                pan_target = np.array([px, py, pz], dtype=float)
                print(f"  Pan object: {px:.3f}, {py:.3f}, {pz:.3f}")

                # Plan arc path to pick from pan
                z_table_for_arc = float(last_pan_xyz[2]-0.065) if last_pan_xyz is not None else 0.05
                tcp_bias = yaw_rotated_tcp_offset(pan_target, SHOULDER_XY_ROBOT2)
                pan_target_biased = pan_target + tcp_bias

                path_pan = plan_adaptive(
                    target_xyz=target2_biased,
                    shoulder_xy=SHOULDER_XY_ROBOT2,
                    back_offset=0.05,
                    up_offset=0.12,
                    f1=0.35, f2=0.3,
                    shape_p=1.8,
                    n_per_segment=8,
                    z_table=z_table_for_arc,
                    clearance=0.00
                )
                phase = Phase.MOVE_TO_PAN_STAGING3

            elif phase == Phase.MOVE_TO_PAN_STAGING3:
                print("[MOVE_TO_PAN_STAGING] Moving to staging near pan")
                goto_xyz(PAN_STAGING_XYZ2, chain, ser, test, label="Pan staging")
                time.sleep(2)
                phase = Phase.APPROACH_PAN_PICKUP_START

            elif phase == Phase.APPROACH_PAN_PICKUP_START:
                print("[MOVE_TO_PAN_PICKUP_START] Moving to pre-approach position")
                goto_xyz(path_pan[0], chain, ser, test, label="Pre-pan pickup")
                phase = Phase.APPROACH_PAN_PICKUP

            elif phase == Phase.APPROACH_PAN_PICKUP:
                print("[APPROACH_PAN_PICKUP] Following arc path to object in pan")
                for (px, py, pz) in path_pan:
                    goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Pan pickup waypoint")
                phase = Phase.GRIP_CLOSE_PAN

            elif phase == Phase.GRIP_CLOSE_PAN:
                print("[GRIP_CLOSE_PAN] Closing gripper on object in pan")
                if not test:
                    send_gripper("CLOSE", ser)
                time.sleep(1.0)
                phase = Phase.LIFT_FROM_PAN_RETURN

            elif phase == Phase.LIFT_FROM_PAN_RETURN:
                print("[LIFT_FROM_PAN_RETURN] Following reverse arc path")
                for (px, py, pz) in path_pan[::-1]:
                    goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Pan retreat waypoint")
                phase = Phase.RETURN_TO_PAN_STAGING_ORG

            elif phase == Phase.RETURN_TO_PAN_STAGING_ORG:
                print("[RETURN_TO_PAN_STAGING] Back to staging")
                goto_xyz(PAN_STAGING_XYZ, chain, ser, test, label="Back to staging")
                phase = Phase.MOVE_TO_ORIGINAL_STAGING

            elif phase == Phase.MOVE_TO_ORIGINAL_STAGING:
                print("[MOVE_TO_ORIGINAL_STAGING] Planning return arc to original position")
                tcp_bias = yaw_rotated_tcp_offset(INITIAL_PICKUP_POS, SHOULDER_XY_ROBOT2)
                original_target_biased = INITIAL_PICKUP_POS + tcp_bias
                path_return = plan_adaptive(
                    target_xyz=original_target_biased,
                    shoulder_xy=SHOULDER_XY_ROBOT2,
                    back_offset=0.1,
                    up_offset=0.09,
                    f1=0.60, f2=0.3,
                    shape_p=1.2,
                    n_per_segment=8,
                    z_table=0.04,
                    clearance=0.010
                )
                if path_return is None or len(path_return) < 2:
                    print("Failed to plan return path")
                    continue
                goto_xyz(path_return[0], chain, ser, test, label="Pre-return")
                phase = Phase.APPROACH_ORIGINAL

            elif phase == Phase.APPROACH_ORIGINAL:
                print("[APPROACH_ORIGINAL] Following arc path to original position")
                for (px, py, pz) in path_return:
                    goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Return waypoint")
                phase = Phase.RELEASE_AT_ORIGINAL

            elif phase == Phase.RELEASE_AT_ORIGINAL:
                print("[RELEASE_AT_ORIGINAL] Releasing object at original position")
                if not test:
                    send_gripper("OPEN", ser)
                time.sleep(1.0)
                phase = Phase.RETREAT_FINAL

            elif phase == Phase.RETREAT_FINAL:
                print("[RETREAT_FINAL] Final retreat: follow last return arc if available, then home")
                # If we have a planned return arc (path_return), follow it in reverse to retreat safely
                try:
                    if 'path_return' in locals() and path_return is not None:
                        for (px, py, pz) in path_return[::-1]:
                            goto_xyz(np.array([px, py, pz]), chain, ser, test, label="Final retreat waypoint")
                except Exception as e:
                    print(f"Warning: failed to follow return path in reverse: {e}")

                # Finally go home
                if ser is not None and not test:
                    send_to_arduino(HOME_US, ser)
                    wait_for_ready(ser)
                else:
                    print(f"Home pulses: {HOME_US}")
                    time.sleep(0.2)
                phase = Phase.DONE

            elif phase == Phase.DONE:
                print("[DONE] All sequences complete. Staying at home.")
                break

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if ser is not None:
            ser.close()

if __name__ == "__main__":
    main(test="--test" in sys.argv)