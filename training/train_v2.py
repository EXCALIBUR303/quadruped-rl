"""
Train the 12-joint robot (robot_v2) to walk and turn.

    caffeinate -i python training/train_v2.py            # resume, or start fresh
    python training/train_v2.py --fresh --seed 7         # reproducible fresh run
    python training/train_v2.py --smoke                  # 30 s pipeline check

Every run writes checkpoints_v2/run_manifest.json recording the git commit,
seed, env config (reward weights + randomization ranges) and PPO hyper-
parameters, so any checkpoint can be traced back to what produced it.
"""

import sys, os, glob, re, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from sim.envs import RobotV2Env
from sim.config import EnvCfg
from training import run_manifest

CHECKPOINT_DIR = "checkpoints_v2"
LOG_DIR        = "logs_v2"
N_ENVS         = 8
TOTAL_STEPS    = 15_000_000

PPO_KWARGS = dict(
    n_steps=2048, batch_size=4096, n_epochs=5,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.005, learning_rate=3e-4, vf_coef=0.5, max_grad_norm=1.0,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def make_env(rank, seed, cfg):
    def _init():
        env = Monitor(RobotV2Env(cfg=cfg))
        env.reset(seed=seed + rank)
        return env
    return _init


def _latest_checkpoint():
    ckpts = glob.glob(f"{CHECKPOINT_DIR}/v2_ppo_*_steps.zip")
    if not ckpts:
        return None
    steps = lambda p: int(re.search(r"v2_ppo_(\d+)_steps", p).group(1))
    latest = max(ckpts, key=steps); n = steps(latest)
    vec = f"{CHECKPOINT_DIR}/v2_ppo_vecnormalize_{n}_steps.pkl"
    return (latest, vec, n) if os.path.exists(vec) else None


def build_envs(seed, cfg, n_envs, vec_cls):
    env = vec_cls([make_env(i, seed, cfg) for i in range(n_envs)])
    eval_env = vec_cls([make_env(1000, seed + 1000, cfg)])
    return env, eval_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore saved checkpoints")
    ap.add_argument("--seed", type=int, default=0, help="master seed (reproducibility)")
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--no-domain-rand", action="store_true",
                    help="disable dynamics randomization (ablation)")
    ap.add_argument("--no-obs-noise", action="store_true",
                    help="disable sensor noise (ablation)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run that validates the whole pipeline in ~30 s")
    args = ap.parse_args()

    cfg = EnvCfg()
    if args.no_domain_rand:
        cfg.domain_rand.enabled = False
    if args.no_obs_noise:
        cfg.obs_noise.enabled = False

    total_steps = args.steps
    n_envs, vec_cls = N_ENVS, SubprocVecEnv
    ckpt_dir, log_dir = CHECKPOINT_DIR, LOG_DIR
    if args.smoke:
        total_steps, n_envs, vec_cls = 4096, 2, DummyVecEnv
        ckpt_dir, log_dir = "checkpoints_smoke", "logs_smoke"
        args.fresh = True

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    set_random_seed(args.seed)

    env, eval_env = build_envs(args.seed, cfg, n_envs, vec_cls)

    ckpt = None if args.fresh else _latest_checkpoint()
    if ckpt:
        mp, vp, n = ckpt
        env = VecNormalize.load(vp, env); env.training = True; env.norm_reward = True
        eval_env = VecNormalize.load(vp, eval_env)
        eval_env.training = False; eval_env.norm_reward = False
        model = PPO.load(mp, env=env, tensorboard_log=log_dir)
        print(f"\n=== RESUMING from {n:,} steps ===\n")
        reset_ts = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                training=False, clip_obs=10.0)
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed,
                    tensorboard_log=log_dir, **PPO_KWARGS)
        print(f"\n=== FRESH run · seed {args.seed} · "
              f"domain_rand={cfg.domain_rand.enabled} obs_noise={cfg.obs_noise.enabled} ===\n")
        reset_ts = True

    manifest_path = os.path.join(ckpt_dir, "run_manifest.json")
    run_manifest.write(manifest_path, run_manifest.collect(
        seed=args.seed, total_steps=total_steps, n_envs=n_envs,
        env_cfg=cfg, ppo_kwargs=PPO_KWARGS,
        extra={"resumed_from": ckpt[2] if ckpt else None, "smoke": args.smoke},
    ))
    print(f"manifest -> {manifest_path}")

    save_freq = max(1, (200_000 if not args.smoke else 2048) // n_envs)
    callbacks = [
        CheckpointCallback(save_freq=save_freq, save_path=ckpt_dir,
                           name_prefix="v2_ppo", save_vecnormalize=True),
        EvalCallback(eval_env, best_model_save_path=ckpt_dir, log_path=log_dir,
                     eval_freq=save_freq, n_eval_episodes=3 if args.smoke else 5,
                     deterministic=True, verbose=1),
    ]
    model.learn(total_timesteps=total_steps, callback=callbacks,
                reset_num_timesteps=reset_ts, progress_bar=not args.smoke)
    model.save(f"{ckpt_dir}/v2_final")
    env.save(f"{ckpt_dir}/vec_normalize.pkl")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
