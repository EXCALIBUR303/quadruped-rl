"""
Gymnasium environment for the 8-servo quadruped (2 joints per leg × 4 legs).

Obs (29) = [torso_z(1), quat(4), linvel(3), angvel(3),
            jnt_pos(8), jnt_vel(8), gait_clock(2)]
Action (8) = normalised motor commands matching real servos 1-to-1.

A periodic "gait clock" drives a trot rhythm: diagonal foot pairs
(FL+RR) and (FR+RL) alternate stance/swing.  The policy sees the clock
(sin φ, cos φ) and is rewarded for matching the trot contact schedule,
which yields a clean, natural four-legged walk instead of a lopsided hop.
"""

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH   = Path(__file__).parent.parent / "assets" / "quadruped.xml"
N_JOINTS   = 8     # 2 per leg × 4 legs
N_FJ_Q     = 7     # freejoint: pos(3) + quat(4)
N_FJ_V     = 6     # freejoint: linvel(3) + angvel(3)
STAND_Z    = 0.24  # target torso height the policy should hold
MIN_Z      = 0.10  # below this → episode over (fell)


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, ctrl_dt=0.02):
        super().__init__()
        self.model    = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data     = mujoco.MjData(self.model)
        self.render_mode  = render_mode
        self._renderer    = None
        self._ctrl_dt     = ctrl_dt
        self._sim_steps   = max(1, round(ctrl_dt / self.model.opt.timestep))  # 4

        # Get keyframe standing pose
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self._stand_qpos = self.model.key_qpos[key_id].copy()

        # Foot geom ids + radius — used for contact detection & clearance reward
        self._foot_gids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
            for leg in ["FL", "FR", "RL", "RR"]
        ]
        self._foot_r        = float(self.model.geom_size[self._foot_gids[0]][0])
        self._max_foot_lift = 0.10   # m — swing-foot clearance cap
        # Trot: diagonal pairs.  group A = FL(0)+RR(3), group B = FR(1)+RL(2).
        self._group_A = (0, 3)

        # Gait clock
        self._gait_period = 0.40     # s per full stride cycle
        self._phase       = 0.0

        n_obs = 1 + 4 + 3 + 3 + N_JOINTS + N_JOINTS + 2  # 29
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32
        )

        self._step         = 0
        self._max_steps    = 1000
        self._target_speed = 0.9   # m/s — forward reward caps here
        self._prev_action  = np.zeros(N_JOINTS, dtype=np.float32)

    # ------------------------------------------------------------------ obs
    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        clock = np.array([np.sin(2 * np.pi * self._phase),
                          np.cos(2 * np.pi * self._phase)])
        return np.concatenate([
            qpos[2:3],           # torso z
            qpos[3:7],           # quaternion
            qvel[0:3],           # linear velocity
            qvel[3:6],           # angular velocity
            qpos[N_FJ_Q:],       # 8 joint angles
            qvel[N_FJ_V:],       # 8 joint velocities
            clock,               # gait clock (sin, cos)
        ]).astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def _foot_contacts(self):
        """Bool array: is each foot's bottom within 2.5 cm of the floor?"""
        return np.array([
            (float(self.data.geom_xpos[g][2]) - self._foot_r) < 0.025
            for g in self._foot_gids
        ])

    # ------------------------------------------------------------------ reward
    def _compute_reward(self, action):
        qpos = self.data.qpos
        qvel = self.data.qvel

        x_vel   = qvel[0]
        y_vel   = qvel[1]
        z       = qpos[2]
        qx, qy  = qpos[4], qpos[5]
        up = 1.0 - 2.0 * float(qx * qx + qy * qy)   # 1 level, 0 tipped 90°

        forward_reward = 2.0 * min(float(x_vel), self._target_speed)
        upright_reward = 1.5 * max(0.0, up)
        height_reward  = 1.0 * max(0.0, 1.0 - abs(z - STAND_Z) / STAND_Z)
        alive          = 0.5

        # ---- Gait-schedule reward: follow the trot rhythm ------------------
        # Group A in stance during first half of the cycle, group B second half.
        # SYMMETRIC: +reward for the right foot state, -penalty for the wrong one,
        # so the policy cannot cheaply freeze a leg planted through its swing.
        contacts  = self._foot_contacts()
        A_stance  = self._phase < 0.5
        gait_reward = 0.0
        for i in range(4):
            in_A          = i in self._group_A
            should_stance = A_stance if in_A else (not A_stance)
            correct       = bool(contacts[i]) == should_stance
            gait_reward  += 0.6 if correct else -0.6   # max +2.4 / min -2.4

        # ---- Penalties (kept light so legs are free to step) --------------
        lateral_penalty    = 0.5 * abs(float(y_vel))
        yaw_penalty        = 0.3 * abs(float(qvel[5]))
        roll_pitch_penalty = 0.3 * float(qvel[3] ** 2 + qvel[4] ** 2)
        smooth_penalty     = 0.01  * float(np.sum((action - self._prev_action) ** 2))
        energy_penalty     = 0.001 * float(np.sum(action ** 2))
        joint_vel_penalty  = 0.0008 * float(np.sum(self.data.qvel[N_FJ_V:] ** 2))

        foot_lift_penalty = 0.0
        for g in self._foot_gids:
            fz = float(self.data.geom_xpos[g][2]) - self._foot_r
            foot_lift_penalty += max(0.0, fz - self._max_foot_lift)
        foot_lift_penalty *= 3.0

        return (forward_reward + upright_reward + height_reward + alive + gait_reward
                - lateral_penalty - yaw_penalty - roll_pitch_penalty
                - smooth_penalty - energy_penalty
                - joint_vel_penalty - foot_lift_penalty)

    # ------------------------------------------------------------------ done
    def _is_terminated(self):
        z      = self.data.qpos[2]
        qx, qy = self.data.qpos[4], self.data.qpos[5]
        up     = 1.0 - 2.0 * float(qx * qx + qy * qy)
        return bool(z < MIN_Z or up < 0.5)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self._stand_qpos.copy()
        self.data.qpos[N_FJ_Q:] += self.np_random.uniform(-0.05, 0.05, size=N_JOINTS)

        mujoco.mj_forward(self.model, self.data)
        self._step        = 0
        self._phase       = float(self.np_random.uniform(0, 1))  # random gait phase
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float32)
        return self._get_obs(), {}

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action

        for _ in range(self._sim_steps):
            mujoco.mj_step(self.model, self.data)

        # advance gait clock
        self._phase = (self._phase + self._ctrl_dt / self._gait_period) % 1.0

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        self._prev_action = action.copy()
        terminated = self._is_terminated()
        self._step += 1
        truncated  = self._step >= self._max_steps

        info = {
            "x_pos":   float(self.data.qpos[0]),
            "torso_z": float(self.data.qpos[2]),
            "x_vel":   float(self.data.qvel[0]),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ render
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="track")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
