"""
Headless evaluation — runs the trained policy without a viewer and prints
objective gait metrics.  Run from the project root:
    python training/evaluate.py
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.envs import QuadrupedEnv

CHECKPOINT_DIR = "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=f"{CHECKPOINT_DIR}/best_model")
    parser.add_argument("--norm",     default=f"{CHECKPOINT_DIR}/vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    raw_env = QuadrupedEnv()
    vec_env = DummyVecEnv([lambda: raw_env])
    if os.path.exists(args.norm):
        vec_env = VecNormalize.load(args.norm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)

    distances, speeds, heights, lengths, falls = [], [], [], [], 0

    for ep in range(args.episodes):
        obs = vec_env.reset()
        ep_heights, steps, fell, last_x = [], 0, False, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            # info[0] holds the pre-reset terminal values on the done step
            ep_heights.append(info[0]["torso_z"])
            last_x = info[0]["x_pos"]
            steps += 1
            if done[0]:
                fell = info[0]["torso_z"] < 0.10
                break
        distances.append(last_x)            # net forward displacement (m)
        speeds.append(np.abs(last_x) / (steps * 0.02))  # avg fwd speed (m/s)
        heights.append(np.mean(ep_heights))
        lengths.append(steps)
        falls += int(fell)
        print(f"  ep {ep+1}: steps={steps:4d}  dist={last_x:+.2f}m  "
              f"mean_z={np.mean(ep_heights):.3f}  {'FELL' if fell else 'survived'}")

    print("\n" + "=" * 50)
    print(f"Episodes:        {args.episodes}")
    print(f"Mean ep length:  {np.mean(lengths):.0f} / 1000 steps")
    print(f"Mean distance:   {np.mean(distances):+.2f} m forward")
    print(f"Mean speed:      {np.mean(speeds):.2f} m/s")
    print(f"Mean torso z:    {np.mean(heights):.3f}  (target 0.24)")
    print(f"Falls:           {falls}/{args.episodes}")
    survived = args.episodes - falls
    walks = np.mean(distances) > 1.0 and survived >= args.episodes * 0.8
    if walks:
        print("VERDICT: Robot walks forward and stays upright. ✓")
    elif survived >= args.episodes * 0.8:
        print("VERDICT: Stays upright but barely moves — tune forward reward.")
    else:
        print("VERDICT: Still unstable — needs more tuning.")
    print("=" * 50)


if __name__ == "__main__":
    main()
