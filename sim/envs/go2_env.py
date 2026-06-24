"""
Unitree Go2 locomotion environment (legged_gym-style), 12-DOF.

Real robot-dog morphology: 3 joints/leg (hip abduction, thigh, calf) — the hip
abduction joint is what makes turning and a stable trot possible.

Control: the policy outputs 12 target joint-angle OFFSETS from the default
stance; a PD controller converts them to motor torques (the standard, stable
way to train legged robots).

Obs (48) = [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
            command(3: vx,vy,yaw), dof_pos-default(12), dof_vel(12), last_action(12)]
Command  = [vx (m/s), vy (m/s, usually 0), yaw (rad/s)]
"""

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH = Path(__file__).parent.parent / "assets" / "go2" / "scene.xml"

# Default standing pose per leg: [hip, thigh, calf]
DEFAULT_POSE = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)
TORQUE_LIMIT = np.array([23.7, 23.7, 45.43] * 4, dtype=np.float64)

KP, KD       = 20.0, 0.5      # PD gains (Go2 legged_gym standard)
ACTION_SCALE = 0.25           # rad, scales policy output to joint targets
DECIMATION   = 10             # sim steps per control step (0.002*10 = 50 Hz)

# Command sampling
VX_RANGE  = (-1.0, 1.5)
YAW_RANGE = (-1.0, 1.0)
STAND_PROB = 0.12

FOOT_GEOMS = ["FL", "FR", "RL", "RR"]


class Go2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data  = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer   = None
        self.dt = self.model.opt.timestep * DECIMATION   # control dt = 0.02 s

        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._home_qpos = self.model.key_qpos[key].copy()

        self._foot_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                           for g in FOOT_GEOMS]
        self._foot_r = float(self.model.geom_size[self._foot_gids[0]][0])

        self.observation_space = spaces.Box(-np.inf, np.inf, (48,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (12,), np.float32)

        self._cmd          = np.zeros(3)        # vx, vy, yaw
        self._last_action  = np.zeros(12)
        self._last_dof_vel = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4, dtype=bool)
        self._step = 0
        self._max_steps = 1000

    # ------------------------------------------------------------ command API
    def set_command(self, vx, yaw=0.0, vy=0.0):
        self._cmd = np.array([vx, vy, yaw], dtype=np.float64)
        self._lock_cmd = True            # don't auto-resample when driven externally

    def _sample_command(self):
        if self.np_random.random() < STAND_PROB:
            return np.zeros(3)
        return np.array([self.np_random.uniform(*VX_RANGE), 0.0,
                         self.np_random.uniform(*YAW_RANGE)])

    # ------------------------------------------------------------ helpers
    def _base_frame(self):
        quat = self.data.qpos[3:7]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
        lin = R.T @ self.data.qvel[0:3]
        ang = R.T @ self.data.qvel[3:6]
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return lin, ang, grav

    def _foot_contacts(self):
        return np.array([
            (float(self.data.geom_xpos[g][2]) - self._foot_r) < 0.02
            for g in self._foot_gids
        ])

    def _get_obs(self):
        lin, ang, grav = self._base_frame()
        dof_pos = self.data.qpos[7:] - DEFAULT_POSE
        dof_vel = self.data.qvel[6:]
        return np.concatenate([
            lin, ang, grav, self._cmd, dof_pos, dof_vel, self._last_action
        ]).astype(np.float32)

    # ------------------------------------------------------------ reward
    def _reward(self, action, torque):
        lin, ang, grav = self._base_frame()
        vx, vy, wz = lin[0], lin[1], ang[2]
        cmd_vx, cmd_vy, cmd_yaw = self._cmd

        # Linear directional-progress reward — pays for velocity TOWARD the
        # command (capped at target, zero when standing). Dense gradient from a
        # standstill that defeats the "stand still and collect partial tracking"
        # local optimum that pure exp kernels fall into with few parallel envs.
        prog_lin = min(vx * np.sign(cmd_vx), abs(cmd_vx)) if abs(cmd_vx) > 0.1 else 0.0
        prog_ang = min(wz * np.sign(cmd_yaw), abs(cmd_yaw)) if abs(cmd_yaw) > 0.1 else 0.0
        progress = 2.5 * prog_lin + 2.0 * prog_ang

        # exp precision bonus on top (locks onto the exact target)
        lin_err = (vx - cmd_vx) ** 2 + (vy - cmd_vy) ** 2
        track_lin = 1.0 * np.exp(-lin_err / 0.25)
        track_ang = 0.5 * np.exp(-(wz - cmd_yaw) ** 2 / 0.25)

        # penalties
        pen_vz       = 2.0  * lin[2] ** 2
        pen_ang_xy   = 0.05 * (ang[0] ** 2 + ang[1] ** 2)
        pen_orient   = 5.0  * (grav[0] ** 2 + grav[1] ** 2)        # 0 when level
        pen_height   = 8.0  * (self.data.qpos[2] - 0.30) ** 2      # stand tall (~0.30 m)
        pen_torque   = 0.0002 * np.sum(torque ** 2)
        pen_actrate  = 0.01 * np.sum((action - self._last_action) ** 2)
        dof_acc      = (self._last_dof_vel - self.data.qvel[6:]) / self.dt
        pen_dof_acc  = 2.5e-7 * np.sum(dof_acc ** 2)

        # feet air time — reward proper swing duration (clean stepping)
        contacts = self._foot_contacts()
        first_contact = contacts & (~self._last_contacts)
        self._feet_air_time += self.dt
        moving = (abs(cmd_vx) + abs(cmd_yaw)) > 0.1
        air_reward = 1.0 * np.sum((self._feet_air_time - 0.3) * first_contact) if moving else 0.0
        self._feet_air_time *= (~contacts)
        self._last_contacts = contacts

        # standing still bonus when commanded to stop
        if not moving:
            stand_bonus = 0.5 * np.exp(-(vx ** 2 + vy ** 2 + wz ** 2) / 0.05)
        else:
            stand_bonus = 0.0

        reward = (progress + track_lin + track_ang + air_reward + stand_bonus
                  - pen_vz - pen_ang_xy - pen_orient - pen_height - pen_torque
                  - pen_actrate - pen_dof_acc)
        return float(reward)

    # ------------------------------------------------------------ done
    def _terminated(self):
        _, _, grav = self._base_frame()
        z = self.data.qpos[2]
        return bool(z < 0.18 or grav[2] > -0.5)   # collapsed or tipped > 60°

    # ------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data,
                                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        self.data.qpos[7:] += self.np_random.uniform(-0.05, 0.05, size=12)
        mujoco.mj_forward(self.model, self.data)

        if options and "command" in options:
            self._cmd = np.array(options["command"], dtype=np.float64)
            self._lock_cmd = True
        else:
            self._cmd = self._sample_command()
            self._lock_cmd = False
        if len(self._cmd) == 2:                        # accept [vx, yaw]
            self._cmd = np.array([self._cmd[0], 0.0, self._cmd[1]])
        self._last_action  = np.zeros(12)
        self._last_dof_vel = self.data.qvel[6:].copy()
        self._feet_air_time = np.zeros(4)
        self._last_contacts = self._foot_contacts()
        self._step = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------ step
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_q = DEFAULT_POSE + ACTION_SCALE * action

        torque = np.zeros(12)
        for _ in range(DECIMATION):
            q  = self.data.qpos[7:]
            qd = self.data.qvel[6:]
            torque = np.clip(KP * (target_q - q) - KD * qd, -TORQUE_LIMIT, TORQUE_LIMIT)
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)

        reward = self._reward(action, torque)
        self._last_action  = action.copy()
        self._last_dof_vel = self.data.qvel[6:].copy()
        self._step += 1

        # resample command mid-episode for better generalisation (training only)
        if self._step % 250 == 0 and not self._lock_cmd:
            self._cmd = self._sample_command()

        obs        = self._get_obs()
        terminated = self._terminated()
        truncated  = self._step >= self._max_steps
        info = {
            "vx": float(self._base_frame()[0][0]),
            "yaw_rate": float(self._base_frame()[1][2]),
            "torso_z": float(self.data.qpos[2]),
            "cmd_vx": float(self._cmd[0]), "cmd_yaw": float(self._cmd[2]),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ render
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
