"""
Gymnasium environment wrapping the MuJoCo quadruped model.
Observation: [torso_z, quat(4), linvel(3), angvel(3), qpos(12), qvel(12)] = 35
Action:      8 normalized motor commands (hip + thigh, per leg)
"""

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH = Path(__file__).parent.parent / "assets" / "quadruped.xml"

# Indices into qpos / qvel for the 12 leg joints (after the 7 freejoint DOF)
# freejoint: qpos[0:7] (pos + quat), qvel[0:6] (linvel + angvel)
N_FREEJOINT_Q = 7
N_FREEJOINT_V = 6
N_JOINTS = 12   # 3 per leg × 4 legs


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, ctrl_dt=0.02):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        # How many sim steps per policy step
        self._sim_steps = max(1, round(ctrl_dt / self.model.opt.timestep))

        n_obs = 1 + 4 + 3 + 3 + N_JOINTS + N_JOINTS   # 35
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Track episode stats
        self._step = 0
        self._max_steps = 1000
        self._prev_x = 0.0

    # ------------------------------------------------------------------
    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        torso_z  = qpos[2:3]              # height
        torso_q  = qpos[3:7]              # orientation quaternion
        lin_vel  = qvel[0:3]              # linear velocity (world frame)
        ang_vel  = qvel[3:6]              # angular velocity
        jnt_pos  = qpos[N_FREEJOINT_Q:]   # 12 joint angles
        jnt_vel  = qvel[N_FREEJOINT_V:]   # 12 joint velocities

        return np.concatenate([
            torso_z, torso_q, lin_vel, ang_vel, jnt_pos, jnt_vel
        ]).astype(np.float32)

    def _compute_reward(self, action):
        x_vel = self.data.qvel[0]           # forward (x) velocity
        z_pos = self.data.qpos[2]           # height of torso

        # Upright: torso quaternion w-component close to 1 → not tilted
        quat_w = self.data.qpos[6]
        upright = quat_w ** 2              # 1 when flat, < 1 when tilted

        forward_reward = 3.0 * x_vel
        upright_reward = 1.0 * upright
        alive_reward   = 0.5               # constant bonus per step
        energy_penalty = 0.001 * np.sum(action ** 2)
        height_penalty = 2.0 * max(0.0, 0.10 - z_pos)  # penalise belly-flopping

        return forward_reward + upright_reward + alive_reward - energy_penalty - height_penalty

    def _is_terminated(self):
        z = self.data.qpos[2]
        return bool(z < 0.08)  # fell over / collapsed

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Small random perturbation so policy learns to recover
        self.data.qpos[N_FREEJOINT_Q:] += self.np_random.uniform(
            -0.05, 0.05, size=N_JOINTS
        )
        self.data.qpos[2] = 0.25   # reset height

        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._prev_x = self.data.qpos[0]
        return self._get_obs(), {}

    def step(self, action):
        # Map 8 policy actions → 8 actuators
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)

        for _ in range(self._sim_steps):
            mujoco.mj_step(self.model, self.data)

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_terminated()
        self._step += 1
        truncated  = self._step >= self._max_steps

        info = {
            "x_pos":  self.data.qpos[0],
            "torso_z": self.data.qpos[2],
            "x_vel":  self.data.qvel[0],
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="track")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
