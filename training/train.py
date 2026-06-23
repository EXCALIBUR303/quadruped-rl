"""
PPO training script.  Run from the project root:
    source .venv/bin/activate
    python training/train.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from sim.envs import QuadrupedEnv

CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"
N_ENVS         = 4          # parallel workers (safe on M-series Mac)
TOTAL_STEPS    = 6_000_000  # speed-control policy (simpler — converges faster)


def make_env(rank: int, seed: int = 0):
    def _init():
        env = QuadrupedEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Vectorised envs for speed
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = SubprocVecEnv([make_env(0, seed=42)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=50_000 // N_ENVS,
            save_path=CHECKPOINT_DIR,
            name_prefix="quadruped_ppo",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=CHECKPOINT_DIR,
            log_path=LOG_DIR,
            eval_freq=50_000 // N_ENVS,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
    ]

    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, progress_bar=True)

    model.save(f"{CHECKPOINT_DIR}/quadruped_final")
    env.save(f"{CHECKPOINT_DIR}/vec_normalize.pkl")
    print("\nTraining complete. Run:  python training/enjoy.py")


if __name__ == "__main__":
    main()
