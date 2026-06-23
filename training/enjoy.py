"""
Watch a trained policy in the MuJoCo viewer.  Run from the project root:
    source .venv/bin/activate
    python training/enjoy.py [--model checkpoints/best_model]
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.envs import QuadrupedEnv

CHECKPOINT_DIR = "checkpoints"
RENDER_FPS = 50   # slow enough to watch comfortably


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{CHECKPOINT_DIR}/best_model")
    parser.add_argument("--norm",  default=f"{CHECKPOINT_DIR}/vec_normalize.pkl")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    raw_env = QuadrupedEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: raw_env])

    if os.path.exists(args.norm):
        vec_env = VecNormalize.load(args.norm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)

    obs = vec_env.reset()
    total_reward = 0.0
    step_dt = 1.0 / RENDER_FPS

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = raw_env.model.body("torso").id
        viewer.cam.distance = 1.2

        print("Viewer open — watch the robot walk! Close the window to quit.")
        for step in range(args.steps):
            t0 = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            viewer.sync()

            # pace to real time so it's watchable
            elapsed = time.time() - t0
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            if not viewer.is_running():
                break

            if done[0]:
                print(f"  Episode ended at step {step}, total reward {total_reward:.1f}")
                obs = vec_env.reset()
                total_reward = 0.0

    print(f"\nDone.  Total reward: {total_reward:.1f}")


if __name__ == "__main__":
    main()
