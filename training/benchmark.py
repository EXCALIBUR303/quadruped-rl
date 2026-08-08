"""
Robustness benchmark — score a policy under perturbed dynamics, not just the
nominal ones it trained on.

`evaluate_v2.py` answers "does it obey commands in perfect conditions?".
This answers the question that actually predicts real-world behaviour: "does it
still work when the floor is slipperier, the battery is sagging, or there is a
payload on its back?" A policy that scores well here has not merely memorized
one exact set of physics.

    python training/benchmark.py                               # current policy
    python training/benchmark.py --model models/final/best_model \
                                --norm  models/final/vec_normalize.pkl
    python training/benchmark.py --compare models/pre_domain_rand
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.config import EnvCfg, KP, KD, TORQUE_LIMIT
from sim.envs import RobotV2Env

# (label, cmd_vx, cmd_yaw)
COMMANDS = [
    ("stand",      0.0,  0.0),
    ("walk slow",  0.3,  0.0),
    ("walk fast",  0.6,  0.0),
    ("turn left",  0.0,  0.6),
    ("turn right", 0.0, -0.6),
    ("walk+turn",  0.4,  0.4),
]

# Held-out stress conditions. These are deliberately at//beyond the edges of
# the training randomization ranges — an honest robustness test, not a replay
# of the training distribution.
CONDITIONS = {
    "nominal":         dict(friction=1.00, mass=0.00, kp=1.00, torque=1.00),
    "ice (fric 0.25)": dict(friction=0.25, mass=0.00, kp=1.00, torque=1.00),
    "grippy (fric 2)": dict(friction=2.00, mass=0.00, kp=1.00, torque=1.00),
    "payload +1.0kg":  dict(friction=1.00, mass=1.00, kp=1.00, torque=1.00),
    "weak motors":     dict(friction=1.00, mass=0.00, kp=0.55, torque=0.60),
    "stiff motors":    dict(friction=1.00, mass=0.00, kp=1.80, torque=1.00),
    "worst case":      dict(friction=0.35, mass=0.70, kp=0.65, torque=0.70),
}


def load_policy(model_path, norm_path, cfg):
    raw = RobotV2Env(cfg=cfg)
    venv = DummyVecEnv([lambda: raw])
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(model_path, env=venv)
    return raw, venv, model


def apply_condition(raw, c):
    """Force a specific dynamics condition, overriding randomization."""
    raw.model.geom_friction[raw._foot_gids, 0] = raw._nominal_fric * c["friction"]
    raw.model.body_mass[raw._torso_bid] = raw._nominal_mass + c["mass"]
    raw._kp = KP * c["kp"]
    raw._kd = KD
    raw._torque_limit = TORQUE_LIMIT * c["torque"]


def run_episode(raw, venv, model, cvx, cyaw, cond, steps=250, seed=None):
    if seed is not None:
        # Seed the ENV's RNG. np.random.seed() does not reach it, so without
        # this the "seeds" only produced uncontrolled episode-to-episode drift
        # and comparisons between policies were not on identical footing.
        raw.reset(seed=seed)
    obs = venv.reset()
    apply_condition(raw, cond)          # after reset, which re-randomizes
    raw.set_command(cvx, cyaw)
    vxs, yaws, hts, fell = [], [], [], False
    for _ in range(steps):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(act)
        apply_condition(raw, cond)
        raw.set_command(cvx, cyaw)
        vxs.append(info[0]["x_vel"]); yaws.append(info[0]["yaw_rate"])
        hts.append(info[0]["torso_z"])
        if done[0]:
            fell = info[0]["torso_z"] < 0.14
            if fell:
                break
    warm = slice(30, None)
    return dict(
        vx=float(np.mean(vxs[warm])) if len(vxs) > 30 else 0.0,
        yaw=float(np.mean(yaws[warm])) if len(yaws) > 30 else 0.0,
        height=float(np.mean(hts[warm])) if len(hts) > 30 else 0.0,
        fell=fell,
    )


def score_policy(model_path, norm_path, label, seeds=2):
    """Return {condition: {tracking_error, fall_rate, height}}."""
    cfg = EnvCfg()
    cfg.domain_rand.enabled = False   # conditions are set explicitly
    cfg.obs_noise.enabled = True      # keep sensor noise: deployment is noisy
    raw, venv, model = load_policy(model_path, norm_path, cfg)

    results = {}
    for cname, cond in CONDITIONS.items():
        errs, falls, heights = [], 0, []
        for cmd_label, cvx, cyaw in COMMANDS:
            for s in range(seeds):
                np.random.seed(s)
                r = run_episode(raw, venv, model, cvx, cyaw, cond, seed=1000 + s)
                errs.append(abs(r["vx"] - cvx) + abs(r["yaw"] - cyaw))
                heights.append(r["height"])
                falls += int(r["fell"])
        n = len(COMMANDS) * seeds
        results[cname] = dict(
            tracking_error=float(np.mean(errs)),
            fall_rate=falls / n,
            height=float(np.mean(heights)),
        )
    raw.close()
    return results


def print_table(name, res):
    print(f"\n  {name}")
    print(f"  {'condition':<18}{'track err':>11}{'falls':>8}{'height':>9}")
    print("  " + "-" * 46)
    for cond, r in res.items():
        print(f"  {cond:<18}{r['tracking_error']:>11.3f}"
              f"{r['fall_rate']*100:>7.0f}%{r['height']:>9.3f}")
    print("  " + "-" * 46)
    print(f"  {'MEAN':<18}{np.mean([r['tracking_error'] for r in res.values()]):>11.3f}"
          f"{np.mean([r['fall_rate'] for r in res.values()])*100:>7.0f}%"
          f"{np.mean([r['height'] for r in res.values()]):>9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints_v2/best_model")
    ap.add_argument("--norm",  default="checkpoints_v2/vec_normalize.pkl")
    ap.add_argument("--compare", default=None,
                    help="directory of a second policy to compare against")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    print("=" * 50)
    print("ROBUSTNESS BENCHMARK  (lower track err / falls = better)")
    print("=" * 50)

    current = score_policy(args.model, args.norm, "current", args.seeds)
    print_table("CURRENT", current)

    if args.compare:
        base = score_policy(os.path.join(args.compare, "best_model"),
                            os.path.join(args.compare, "vec_normalize.pkl"),
                            "baseline", args.seeds)
        print_table(f"BASELINE ({args.compare})", base)

        print("\n  DELTA (current - baseline; negative = improvement)")
        print("  " + "-" * 46)
        for cond in CONDITIONS:
            de = current[cond]["tracking_error"] - base[cond]["tracking_error"]
            df = (current[cond]["fall_rate"] - base[cond]["fall_rate"]) * 100
            print(f"  {cond:<18}{de:>+11.3f}{df:>+7.0f}%")
        me = (np.mean([r["tracking_error"] for r in current.values()])
              - np.mean([r["tracking_error"] for r in base.values()]))
        mf = (np.mean([r["fall_rate"] for r in current.values()])
              - np.mean([r["fall_rate"] for r in base.values()])) * 100
        print("  " + "-" * 46)
        print(f"  {'MEAN':<18}{me:>+11.3f}{mf:>+7.0f}%")
        print(f"\n  VERDICT: {'more robust ✓' if me < 0 or mf < 0 else 'no robustness gain'}")


if __name__ == "__main__":
    main()
