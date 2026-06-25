"""Command-tracking eval for the 12-joint robot — speed AND turning."""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sim.envs import RobotV2Env

CKPT = "checkpoints_v2"
TESTS = [
    ("stand still",   0.0,  0.0),
    ("walk slow",     0.3,  0.0),
    ("walk fast",     0.6,  0.0),
    ("turn left",     0.0,  0.6),
    ("turn right",    0.0, -0.6),
    ("walk + turn",   0.4,  0.4),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{CKPT}/best_model")
    ap.add_argument("--norm",  default=f"{CKPT}/vec_normalize.pkl")
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()

    import mujoco
    raw = RobotV2Env()
    venv = DummyVecEnv([lambda: raw])
    if os.path.exists(args.norm):
        venv = VecNormalize.load(args.norm, venv); venv.training = False; venv.norm_reward = False
    model = PPO.load(args.model, env=venv)
    feet = [mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot")
            for l in ["FL", "FR", "RL", "RR"]]
    fr = raw.model.geom_size[feet[0]][0]

    # gait quality is checked too, so a collapsed / sliding robot can't "pass"
    print(f"\n{'command':<13}{'asked':>15}{'measured':>17}{'height':>8}{'slip':>7}{'':>7}")
    print("-" * 67)
    passed = 0
    for label, cvx, cyaw in TESTS:
        venv.env_method("set_command", cvx, cyaw)
        obs = venv.reset(); raw.set_command(cvx, cyaw)
        vxs, yaws, hts, slips, fell = [], [], [], [], False
        prev = np.array([raw.data.geom_xpos[g][:2].copy() for g in feet])
        for _ in range(args.steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(a); raw.set_command(cvx, cyaw)
            vxs.append(info[0]["x_vel"]); yaws.append(info[0]["yaw_rate"]); hts.append(info[0]["torso_z"])
            cur = np.array([raw.data.geom_xpos[g][:2].copy() for g in feet])
            for i, g in enumerate(feet):
                if raw.data.geom_xpos[g][2] - fr < 0.025:
                    slips.append(np.linalg.norm(cur[i] - prev[i]) / 0.02)
            prev = cur
            if done[0]:
                fell = info[0]["torso_z"] < 0.14; break
        mvx, myaw, mh = np.mean(vxs[30:]), np.mean(yaws[30:]), np.mean(hts[30:])
        mslip = np.mean(slips) if slips else 0.0
        track_ok  = abs(mvx - cvx) < 0.25 and abs(myaw - cyaw) < 0.35
        gait_ok   = mh > 0.20 and mslip < 0.25       # upright AND stepping (not sliding)
        ok = (not fell) and track_ok and gait_ok
        passed += int(ok)
        print(f"{label:<13}{f'vx={cvx:+.1f} yaw={cyaw:+.1f}':>15}"
              f"{f'vx={mvx:+.2f} yaw={myaw:+.2f}':>17}{mh:>8.2f}{mslip:>7.2f}"
              f"{'  PASS' if ok else '  FAIL':>7}")
    print("-" * 67)
    print("(height>0.20 = upright, slip<0.25 = stepping not sliding)")
    print(f"Passed {passed}/{len(TESTS)}")
    print("VERDICT: walks + turns with a real gait. ✓" if passed >= len(TESTS) - 1
          else "VERDICT: needs more training.")


if __name__ == "__main__":
    main()
