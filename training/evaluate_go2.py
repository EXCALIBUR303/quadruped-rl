"""
Command-tracking evaluation for the Unitree Go2 policy — checks it obeys
forward speed AND turn commands.  Run from the project root:
    python training/evaluate_go2.py
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.envs import Go2Env

CKPT = "checkpoints_go2"

# (label, cmd_vx, cmd_yaw)
TESTS = [
    ("stand still",   0.0,  0.0),
    ("walk slow",     0.5,  0.0),
    ("walk fast",     1.2,  0.0),
    ("walk backward",-0.5,  0.0),
    ("turn left",     0.0,  0.8),
    ("turn right",    0.0, -0.8),
    ("forward + left",0.6,  0.6),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{CKPT}/best_model")
    ap.add_argument("--norm",  default=f"{CKPT}/vec_normalize.pkl")
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()

    raw = Go2Env()
    venv = DummyVecEnv([lambda: raw])
    if os.path.exists(args.norm):
        venv = VecNormalize.load(args.norm, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(args.model, env=venv)

    print(f"\n{'command':<16}{'asked':>16}{'measured':>18}{'result':>8}")
    print("-" * 60)
    passed = 0
    for label, cvx, cyaw in TESTS:
        venv.env_method("set_command", cvx, cyaw)
        obs = venv.reset()
        raw.set_command(cvx, cyaw)
        vxs, yaws, fell = [], [], False
        for _ in range(args.steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(a)
            raw.set_command(cvx, cyaw)
            vxs.append(info[0]["vx"]); yaws.append(info[0]["yaw_rate"])
            if done[0]:
                fell = info[0]["torso_z"] < 0.18
                break
        mvx, myaw = np.mean(vxs[30:]), np.mean(yaws[30:])
        ok = (not fell) and abs(mvx - cvx) < 0.3 and abs(myaw - cyaw) < 0.35
        passed += int(ok)
        print(f"{label:<16}{f'vx={cvx:+.1f} yaw={cyaw:+.1f}':>16}"
              f"{f'vx={mvx:+.2f} yaw={myaw:+.2f}':>18}{'  PASS' if ok else '  FAIL':>8}")
    print("-" * 60)
    print(f"Passed {passed}/{len(TESTS)}")
    print("VERDICT: Go2 follows speed + turn commands. ✓" if passed >= len(TESTS) - 1
          else "VERDICT: needs more training / tuning.")


if __name__ == "__main__":
    main()
