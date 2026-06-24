"""
Train the Unitree Go2 (12-DOF) to follow velocity + turn commands.
    source .venv/bin/activate
    python training/train_go2.py
"""

import sys, os, glob, re, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from sim.envs import Go2Env

CHECKPOINT_DIR = "checkpoints_go2"
LOG_DIR        = "logs_go2"
N_ENVS         = 8
TOTAL_STEPS    = 25_000_000   # 12-DOF locomotion needs lots of experience


def make_env(rank, seed=0):
    def _init():
        env = Monitor(Go2Env())
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def _latest_checkpoint():
    """Return (model_path, vecnorm_path, steps) for the newest saved checkpoint."""
    ckpts = glob.glob(f"{CHECKPOINT_DIR}/go2_ppo_*_steps.zip")
    if not ckpts:
        return None
    def steps(p):
        m = re.search(r"go2_ppo_(\d+)_steps", p)
        return int(m.group(1)) if m else 0
    latest = max(ckpts, key=steps)
    n = steps(latest)
    vec = f"{CHECKPOINT_DIR}/go2_ppo_vecnormalize_{n}_steps.pkl"
    return (latest, vec, n) if os.path.exists(vec) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="ignore saved checkpoints and start over")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    eval_env = SubprocVecEnv([make_env(100, seed=999)])

    ckpt = None if args.fresh else _latest_checkpoint()
    if ckpt:
        model_path, vec_path, n_done = ckpt
        env = VecNormalize.load(vec_path, env)         # continue normalisation stats
        env.training = True; env.norm_reward = True
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False; eval_env.norm_reward = False
        model = PPO.load(model_path, env=env, tensorboard_log=LOG_DIR)
        print(f"\n=== RESUMING from {n_done:,} steps "
              f"({TOTAL_STEPS - n_done:,} remaining) ===\n")
        reset_timesteps = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                training=False, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", env,
            n_steps=1024, batch_size=2048, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, learning_rate=3e-4, vf_coef=0.5, max_grad_norm=1.0,
            verbose=1, tensorboard_log=LOG_DIR,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
        )
        print("\n=== Starting FRESH ===\n")
        reset_timesteps = True

    callbacks = [
        CheckpointCallback(save_freq=200_000 // N_ENVS, save_path=CHECKPOINT_DIR,
                           name_prefix="go2_ppo", save_vecnormalize=True),
        EvalCallback(eval_env, best_model_save_path=CHECKPOINT_DIR, log_path=LOG_DIR,
                     eval_freq=200_000 // N_ENVS, n_eval_episodes=5,
                     deterministic=True, verbose=1),
    ]

    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks,
                reset_num_timesteps=reset_timesteps, progress_bar=True)
    model.save(f"{CHECKPOINT_DIR}/go2_final")
    env.save(f"{CHECKPOINT_DIR}/vec_normalize.pkl")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
