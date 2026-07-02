"""
Render the full showcase video: walking, turning, tricks, and autonomous fetch.
    python scripts/render_walk.py
Produces robot_demo.mp4 in the project root.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brain"))

import numpy as np
import mujoco
import imageio
from controller import RobotController

FPS = 50


def main():
    ctrl = RobotController()
    m, d = ctrl.raw.model, ctrl.raw.data

    renderer = mujoco.Renderer(m, height=720, width=1280)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance = 1.7; cam.azimuth = 125; cam.elevation = -16

    frames = []

    def record(steps):
        for _ in range(steps):
            ctrl.step()
            cam.lookat = d.qpos[:3].copy()
            renderer.update_scene(d, cam)
            frames.append(renderer.render())

    def record_while(mode, limit=1200):
        n = 0
        while ctrl.mode == mode and n < limit:
            ctrl.step()
            cam.lookat = d.qpos[:3].copy()
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
            n += 1

    # --- choreography ---------------------------------------------------
    print("scene 1: walk + turns")
    ctrl.set_command(0.45, 0.0); record(int(3.0 * FPS))
    ctrl.set_command(0.0, 0.6);  record(int(2.5 * FPS))
    ctrl.set_command(0.45, 0.0); record(int(2.0 * FPS))
    ctrl.set_command(0.0, 0.0);  record(int(0.8 * FPS))

    print("scene 2: tricks")
    for trick in ["sit", "wave", "bow", "dance"]:
        ctrl.do_trick(trick)
        record_while("TRICK")
        record(int(0.4 * FPS))

    print("scene 3: fetch")
    ctrl.fetch()
    record_while("NAVIGATE")
    record_while("TRICK")            # celebration wiggle
    record(int(1.0 * FPS))

    print("scene 4: goodbye sit")
    ctrl.do_trick("sit")
    record_while("TRICK")

    out = "robot_demo.mp4"
    imageio.mimsave(out, frames, fps=FPS, quality=8)
    print(f"\nSaved {out}  ({len(frames)} frames, {len(frames)/FPS:.0f}s)")


if __name__ == "__main__":
    main()
