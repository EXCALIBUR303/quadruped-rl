"""
Render a video of the trained 12-joint robot walking and turning.
    python scripts/render_walk.py
Produces robot_demo.mp4 in the project root.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mujoco
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.envs import RobotV2Env

CKPT = "checkpoints_v2"
FPS  = 50

# (label, vx, yaw, seconds)
SCRIPT = [
    ("walk forward", 0.6,  0.0, 4.0),
    ("turn left",    0.0,  0.6, 3.0),
    ("walk fast",    0.8,  0.0, 4.0),
    ("turn right",   0.0, -0.6, 3.0),
    ("walk + curve", 0.5,  0.4, 3.0),
    ("stop",         0.0,  0.0, 2.0),
]


def main():
    raw = RobotV2Env()
    venv = DummyVecEnv([lambda: raw])
    venv = VecNormalize.load(f"{CKPT}/vec_normalize.pkl", venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(f"{CKPT}/best_model", env=venv)

    renderer = mujoco.Renderer(raw.model, height=720, width=1280)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(raw.model, cam)
    cam.distance = 1.6; cam.azimuth = 130; cam.elevation = -18

    obs = venv.reset()
    frames = []
    for label, vx, yaw, secs in SCRIPT:
        raw.set_command(vx, yaw)
        for _ in range(int(secs * FPS)):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(a)
            raw.set_command(vx, yaw)
            # camera follows the robot
            cam.lookat = raw.data.qpos[:3].copy()
            renderer.update_scene(raw.data, cam)
            frames.append(renderer.render())
            if done[0]:
                obs = venv.reset(); raw.set_command(vx, yaw)
        print(f"  rendered: {label}")

    out = "robot_demo.mp4"
    imageio.mimsave(out, frames, fps=FPS, quality=8)
    print(f"\nSaved {out}  ({len(frames)} frames, {len(frames)/FPS:.0f}s)")


if __name__ == "__main__":
    main()
