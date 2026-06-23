"""
Command-tracking evaluation — checks the policy obeys specific velocity
commands (stand, slow, fast, turn).  Run from the project root:
    python training/evaluate.py
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.envs import QuadrupedEnv

CHECKPOINT_DIR = "checkpoints"

# (label, cmd_vx, cmd_yaw)
TEST_COMMANDS = [
    ("stand still",   0.0,  0.0),
    ("walk slow",     0.3,  0.0),
    ("walk fast",     0.6,  0.0),
    ("walk backward",-0.3,  0.0),
    ("turn left",     0.0,  0.5),   # turn in place (what the curriculum trains)
    ("turn right",    0.0, -0.5),
    ("walk+turn L",   0.3,  0.3),   # combined (the hard case)
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{CHECKPOINT_DIR}/best_model")
    parser.add_argument("--norm",  default=f"{CHECKPOINT_DIR}/vec_normalize.pkl")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    raw = QuadrupedEnv()
    venv = DummyVecEnv([lambda: raw])
    if os.path.exists(args.norm):
        venv = VecNormalize.load(args.norm, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(args.model, env=venv)

    print(f"\n{'command':<14}{'asked':>16}{'measured':>18}{'result':>10}")
    print("-" * 60)
    passes = 0
    for label, cvx, cyaw in TEST_COMMANDS:
        venv.env_method("set_command", cvx, cyaw)
        obs = venv.reset()
        raw.set_command(cvx, cyaw)             # ensure command set post-reset
        vxs, yaws, fell = [], [], False
        for _ in range(args.steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(a)
            # re-assert command each step (reset() would otherwise randomise it)
            raw.set_command(cvx, cyaw)
            vxs.append(info[0]["x_vel"])
            yaws.append(info[0]["yaw_rate"])
            if done[0]:
                fell = info[0]["torso_z"] < 0.10
                break
        mvx, myaw = np.mean(vxs[20:]), np.mean(yaws[20:])
        vx_ok  = abs(mvx - cvx) < 0.25
        yaw_ok = abs(myaw - cyaw) < 0.35
        ok = (not fell) and vx_ok and yaw_ok
        passes += int(ok)
        print(f"{label:<14}{f'vx={cvx:+.1f} yaw={cyaw:+.1f}':>16}"
              f"{f'vx={mvx:+.2f} yaw={myaw:+.2f}':>18}"
              f"{'  PASS' if ok else '  FAIL':>10}")

    print("-" * 60)
    print(f"Passed {passes}/{len(TEST_COMMANDS)} command checks")
    if passes >= len(TEST_COMMANDS) - 1:
        print("VERDICT: Policy follows velocity commands. ✓")
    else:
        print("VERDICT: Command tracking weak — needs tuning / more training.")


if __name__ == "__main__":
    main()
