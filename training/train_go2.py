"""
Train the Unitree Go2 (12-DOF) to follow velocity + turn commands.
    source .venv/bin/activate
    python training/train_go2.py
"""

import sys, os
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


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = SubprocVecEnv([make_env(100, seed=999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=1024,            # 8 envs × 1024 = 8192 samples / update
        batch_size=2048,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
        vf_coef=0.5,
        max_grad_norm=1.0,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[512, 256, 128]),
    )

    callbacks = [
        CheckpointCallback(save_freq=200_000 // N_ENVS, save_path=CHECKPOINT_DIR,
                           name_prefix="go2_ppo", save_vecnormalize=True),
        EvalCallback(eval_env, best_model_save_path=CHECKPOINT_DIR, log_path=LOG_DIR,
                     eval_freq=200_000 // N_ENVS, n_eval_episodes=5,
                     deterministic=True, verbose=1),
    ]

    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, progress_bar=True)
    model.save(f"{CHECKPOINT_DIR}/go2_final")
    env.save(f"{CHECKPOINT_DIR}/vec_normalize.pkl")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
