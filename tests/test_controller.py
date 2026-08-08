"""Controller mode-machine integration tests.

These load a real policy and run real physics (headless — the viewer is never
opened). They guard the behaviours that broke in the past: the episode-
truncation teleport, tricks failing to hand control back to the walker, and
concurrent mode switching from the voice thread.
"""

import threading
import numpy as np
import pytest

from conftest import requires_checkpoint

pytestmark = requires_checkpoint

ACT = 12


# ------------------------------------------------------------------- basics
def test_starts_in_policy_mode_and_stationary(controller):
    assert controller.mode == "POLICY"
    assert controller.speed == 0.0 and controller.turn == 0.0


def _xy(controller):
    return np.array(controller.raw.data.qpos[:2], dtype=float)


def _distance_walked(controller, steps, vx, yaw=0.0):
    """Path length in the xy plane.

    Deliberately heading-agnostic: the policy is body-frame, so it walks in
    *its own* forward direction. Asserting on world-frame x would silently pass
    or fail depending on which way the robot happens to be facing — the exact
    world-frame assumption that caused the original 'only walks east' bug.
    """
    controller.set_command(vx, yaw)
    prev = _xy(controller)
    travelled, jumps = 0.0, 0
    for _ in range(steps):
        controller.step()
        cur = _xy(controller)
        d = float(np.linalg.norm(cur - prev))
        if d > 0.25:                          # 0.25 m in one 20 ms step = teleport
            jumps += 1
        travelled += d
        prev = cur
    return travelled, jumps


def test_walk_command_moves_the_robot_forward(controller):
    dist, _ = _distance_walked(controller, 150, 0.4)
    assert dist > 0.3, f"robot did not walk (travelled {dist:.2f} m)"
    assert float(controller.raw.data.qpos[2]) > 0.20, "robot collapsed while walking"


def test_no_teleport_during_a_long_run(controller):
    """Regression: episode truncation used to auto-reset the sim mid-demo,
    yanking the robot back to the origin roughly every 20 s."""
    dist, jumps = _distance_walked(controller, 1500, 0.3)   # 30 s
    assert jumps == 0, f"{jumps} discontinuous jumps — the robot teleported"
    assert dist > 3.0, f"robot barely moved over 30 s (travelled {dist:.2f} m)"


# -------------------------------------------------------------------- tricks
def test_trick_runs_and_returns_control_to_the_policy(controller):
    assert controller.do_trick("sit") is True
    assert controller.mode == "TRICK"
    for _ in range(2000):
        controller.step()
        if controller.mode != "TRICK":
            break
    assert controller.mode == "POLICY", "trick never handed control back"
    assert float(controller.raw.data.qpos[2]) > 0.22, "robot left collapsed after trick"


def test_policy_still_walks_after_a_trick(controller):
    """Guards the observation re-normalization when leaving direct PD control."""
    controller.do_trick("wiggle")
    while controller.mode == "TRICK":
        controller.step()
    dist, _ = _distance_walked(controller, 150, 0.4)
    assert dist > 0.25, f"walking broke after a trick (travelled {dist:.2f} m)"


def test_unknown_trick_is_rejected_without_changing_mode(controller):
    controller.set_command(0.0, 0.0)
    assert controller.do_trick("moonwalk") is False
    assert controller.mode == "POLICY"


# --------------------------------------------------------------- navigation
def test_fetch_reaches_the_ball_and_celebrates(controller):
    events = []
    controller.on_event = lambda e: events.append(e)
    controller.set_command(0.0, 0.0)
    controller.raw.data.mocap_pos[controller._ball_mocap] = [1.8, 0.6, 0.05]
    controller.mode = "NAVIGATE"
    for _ in range(3000):
        controller.step()
        if "fetched" in events:
            break
    assert "fetched" in events, "fetch never reached the ball"


def test_navigation_avoids_the_obstacles(controller):
    """Steering must keep clear of the crates, not plough through them."""
    controller.set_command(0.0, 0.0)
    controller.raw.data.mocap_pos[controller._ball_mocap] = [2.7, 1.65, 0.05]
    controller.mode = "NAVIGATE"
    worst = 9.9
    for _ in range(3000):
        controller.step()
        p = controller.raw.data.qpos[:2]
        for ox, oy, r in controller._obstacles:
            worst = min(worst, float(np.hypot(p[0] - ox, p[1] - oy)) - r)
        if controller.mode != "NAVIGATE":
            break
    assert worst > 0.0, f"robot entered a crate (clearance {worst:.2f} m)"


def test_obstacles_were_loaded_from_the_model(controller):
    assert len(controller._obstacles) >= 4, "crate obstacles not parsed from the XML"


# ------------------------------------------------------------ thread safety
def test_concurrent_mode_switching_does_not_crash(controller):
    """The voice thread changes modes while the physics thread steps.

    Regression: step() read _trick_seq and _trick_i non-atomically, so a mode
    switch landing between them raised IndexError.
    """
    errors = []
    stop = threading.Event()

    def stepper():
        try:
            while not stop.is_set():
                controller.step()
        except Exception as exc:              # pragma: no cover - failure path
            errors.append(exc)

    t = threading.Thread(target=stepper, daemon=True)
    t.start()
    try:
        for i in range(120):
            controller.do_trick(["sit", "wiggle", "bow"][i % 3])
            controller.set_command(0.3, 0.2)
            controller.spin(seconds=0.4)
            controller.fetch()
    finally:
        stop.set()
        t.join(timeout=10)
    assert not errors, f"race condition surfaced: {errors[:3]}"
