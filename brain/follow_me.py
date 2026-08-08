"""
FOLLOW ME — drive the robot with your webcam.

Hold up a BRIGHT ORANGE object (a tennis/table-tennis ball, orange sticky note,
or your phone showing a solid orange screen). The robot mirrors it:

    move it LEFT / RIGHT  -> robot turns left / right
    bring it CLOSER (big) -> robot stops (it "reached" you)
    hold it FAR (small)   -> robot walks toward you
    hide it               -> robot stops

Run (macOS needs mjpython for the MuJoCo viewer):
    mjpython brain/follow_me.py
The first run asks for camera permission — click Allow, then re-run if needed.
Press q in the camera window to quit.
"""

import sys, os, threading, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from controller import RobotController

# HSV range for a bright orange object (tweak SAT/VAL down if your room is dim)
HSV_LO = np.array([5, 120, 120])
HSV_HI = np.array([25, 255, 255])
MIN_AREA_FRAC   = 0.001   # ignore blobs smaller than this fraction of the frame
CLOSE_AREA_FRAC = 0.06    # blob this big = "you're close" -> stop


def detect_target(frame_bgr):
    """Return (cx_norm in [-1,1], area_frac) of the largest orange blob, or None."""
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h, w = frame_bgr.shape[:2]
    if area < MIN_AREA_FRAC * (h * w):
        return None
    M = cv2.moments(c)
    cx = M["m10"] / (M["m00"] + 1e-9)
    return (2.0 * cx / w - 1.0), (area / (h * w))


def command_from_target(target):
    """Map detection -> (vx, yaw). Mirrored so moving the object left turns left."""
    if target is None:
        return 0.0, 0.0
    cx, area = target
    yaw = float(np.clip(-1.0 * cx, -0.6, 0.6))          # object left -> turn left
    if area > CLOSE_AREA_FRAC:
        vx = 0.0                                        # reached you
    else:
        vx = float(np.clip(0.55 * (1.0 - area / CLOSE_AREA_FRAC), 0.25, 0.55))
    if abs(cx) > 0.55:                                  # mostly turning
        vx = 0.0
    return vx, yaw


def camera_loop(ctrl):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("!! could not open the webcam.\n"
              "   Grant access: System Settings → Privacy & Security → Camera "
              "→ enable Terminal, then re-run.\n"
              "   Or drive the robot by keyboard: mjpython brain/controller.py")
        os._exit(1)
    print("camera on — hold up something bright orange")
    lost_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)                      # mirror view
        target = detect_target(frame)
        if target is None:
            lost_frames += 1
            if lost_frames == 10:                       # ~0.3 s without target
                ctrl.set_command(0.0, 0.0)
        else:
            lost_frames = 0
            vx, yaw = command_from_target(target)
            ctrl.set_command(vx, yaw)
            cx, area = target
            h, w = frame.shape[:2]
            cv2.circle(frame, (int((cx + 1) / 2 * w), h // 2),
                       max(8, int(np.sqrt(area * h * w / 3.14))), (0, 165, 255), 3)
            cv2.putText(frame, f"vx={vx:.2f} yaw={yaw:+.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("follow-me (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            os._exit(0)


if __name__ == "__main__":
    import preflight

    ok_viewer, viewer_msg = preflight.check_viewer_runtime()
    if not ok_viewer:
        print(viewer_msg)
        sys.exit(1)
    ok_cam, cam_msg = preflight.check_camera()
    print(("  ✓ " if ok_cam else "  ! ") + cam_msg)
    if not ok_cam:
        sys.exit(1)

    print("loading policy…")
    try:
        ctrl = RobotController()
    except preflight.PreflightError as exc:
        print(exc)
        sys.exit(1)
    threading.Thread(target=lambda: camera_loop(ctrl), daemon=True).start()
    ctrl.run_viewer()
