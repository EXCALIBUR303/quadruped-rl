"""
Environment for the 12-joint robot (robot_v2.xml), with PD POSITION CONTROL.

Previous version used raw torque control → the policy slammed actuators to
their limits (99% saturation), the body collapsed, and the feet slid instead
of stepping. The fix (standard for legged-robot RL, e.g. legged_gym): the
policy outputs TARGET joint-angle offsets from the standing pose, and a PD
loop converts them to smooth motor torques. This yields a stable, stepping
gait and an upright posture.

Obs (51) = [torso_z(1), quat(4), linvel(3), angvel(3),
            (jnt_pos - default)(12), jnt_vel(12), last_action(12),
            gait_clock(2), command(2)]
Action (12) = target joint-angle offsets (scaled). Command = [vx, yaw].
"""

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH = Path(__file__).parent.parent / "assets" / "robot_v2.xml"
N_JOINTS = 12
N_FJ_Q   = 7
N_FJ_V   = 6

# Standing pose per leg [hip, shoulder, knee] — the PD target at action=0.
DEFAULT_POSE  = np.array([0.0, 0.63, -1.25] * 4)
TORQUE_LIMIT  = np.array([8.0, 12.0, 14.0] * 4)
KP, KD        = 25.0, 0.7        # PD gains
ACTION_SCALE  = 0.25             # rad — policy output → target offset
STAND_HEIGHT  = 0.27             # target torso height
MIN_Z         = 0.14             # below this → fallen

VX_RANGE  = (-0.3, 0.8)
YAW_RANGE = (-0.8, 0.8)


class RobotV2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, ctrl_dt=0.02):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data  = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer   = None
        self._ctrl_dt    = ctrl_dt
        self._sim_steps  = max(1, round(ctrl_dt / self.model.opt.timestep))

        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self._stand_qpos = self.model.key_qpos[key].copy()

        self._foot_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot")
                           for l in ["FL", "FR", "RL", "RR"]]
        self._foot_r        = float(self.model.geom_size[self._foot_gids[0]][0])
        self._max_foot_lift = 0.12
        self._group_A       = (0, 3)

        self._freq_min, self._freq_max = 2.2, 4.2
        self._phase = 0.0
        self._cmd   = np.zeros(2)
        self._last_action  = np.zeros(N_JOINTS)
        self._prev_foot_xy = np.zeros((4, 2))

        n_obs = 1 + 4 + 3 + 3 + N_JOINTS + N_JOINTS + N_JOINTS + 2 + 2   # 51
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)

        self._step = 0
        self._max_steps = 1000

    # ----------------------------------------------------------- command API
    def set_command(self, vx, yaw=0.0):
        self._cmd = np.array([vx, yaw], dtype=np.float64)

    def _sample_command(self):
        r = self.np_random.random()
        if r < 0.12:
            return np.zeros(2)
        elif r < 0.55:
            return np.array([self.np_random.uniform(0.2, 0.8), 0.0])
        elif r < 0.80:
            return np.array([0.0, self.np_random.uniform(*YAW_RANGE)])
        else:
            return np.array([self.np_random.uniform(0.2, 0.5),
                             self.np_random.uniform(-0.6, 0.6)])

    # ----------------------------------------------------------- helpers
    def _foot_xy_z(self):
        xy = np.array([self.data.geom_xpos[g][:2] for g in self._foot_gids])
        z  = np.array([self.data.geom_xpos[g][2] - self._foot_r for g in self._foot_gids])
        return xy, z

    def _get_obs(self):
        qpos, qvel = self.data.qpos, self.data.qvel
        clock = np.array([np.sin(2*np.pi*self._phase), np.cos(2*np.pi*self._phase)])
        return np.concatenate([
            qpos[2:3], qpos[3:7], qvel[0:3], qvel[3:6],
            qpos[N_FJ_Q:] - DEFAULT_POSE, qvel[N_FJ_V:], self._last_action,
            clock, self._cmd,
        ]).astype(np.float32)

    # ----------------------------------------------------------- reward
    def _compute_reward(self, action):
        qpos, qvel = self.data.qpos, self.data.qvel
        vx, vy   = float(qvel[0]), float(qvel[1])
        yaw_rate = float(qvel[5])
        z        = qpos[2]
        qx, qy   = qpos[4], qpos[5]
        up = 1.0 - 2.0 * float(qx*qx + qy*qy)

        cmd_vx, cmd_yaw = float(self._cmd[0]), float(self._cmd[1])
        moving = (abs(cmd_vx) > 0.1) or (abs(cmd_yaw) > 0.1)

        prog_vx  = min(vx * np.sign(cmd_vx), abs(cmd_vx)) if abs(cmd_vx) > 0.1 else 0.0
        prog_yaw = min(yaw_rate * np.sign(cmd_yaw), abs(cmd_yaw)) if abs(cmd_yaw) > 0.1 else 0.0
        lin_track = np.exp(-4.0 * (vx - cmd_vx) ** 2)
        yaw_trk   = np.exp(-4.0 * (yaw_rate - cmd_yaw) ** 2)
        track = 4.0*prog_vx + 1.5*lin_track + 3.0*prog_yaw + 1.0*yaw_trk

        # STRONG posture rewards — stop the collapse
        height_reward  = 1.5 * np.exp(-((z - STAND_HEIGHT) / 0.06) ** 2)
        upright_reward = 1.0 * max(0.0, up)
        alive          = 0.3

        # foot contacts + SLIP (penalise dragging) ---------------------------
        foot_xy, foot_z = self._foot_xy_z()
        contacts = foot_z < 0.025
        slip = 0.0
        for i in range(4):
            if contacts[i]:
                slip += np.linalg.norm(foot_xy[i] - self._prev_foot_xy[i]) / self._ctrl_dt
        self._prev_foot_xy = foot_xy
        slip_penalty = 0.6 * slip

        # gait clock (trot when moving / plant when stopped)
        if moving:
            A_stance = self._phase < 0.5
            gait = 0.0
            for i in range(4):
                should = (A_stance if i in self._group_A else not A_stance)
                gait += 0.4 if (bool(contacts[i]) == should) else -0.4
        else:
            gait = 0.5 * (int(contacts.sum()) - 2)

        lateral_penalty   = 0.3 * abs(vy) * (1.0 - min(1.0, abs(cmd_yaw)/0.3))
        yaw_drift_penalty = 1.0 * abs(yaw_rate) if abs(cmd_yaw) < 0.1 else 0.0
        stillness_penalty = (3.0 * (abs(vx) + abs(vy) + abs(yaw_rate)) if not moving else 0.0)
        roll_pitch_pen    = 0.3 * float(qvel[3]**2 + qvel[4]**2)
        action_rate_pen   = 0.05 * float(np.sum((action - self._last_action) ** 2))
        jvel_pen          = 0.0002 * float(np.sum(qvel[N_FJ_V:] ** 2))

        foot_lift_pen = 0.0
        for i in range(4):
            foot_lift_pen += max(0.0, foot_z[i] - self._max_foot_lift)
        foot_lift_pen *= 1.5

        return (track + height_reward + upright_reward + alive + gait
                - slip_penalty - lateral_penalty - yaw_drift_penalty
                - stillness_penalty - roll_pitch_pen - action_rate_pen
                - jvel_pen - foot_lift_pen)

    def _is_terminated(self):
        z = self.data.qpos[2]
        qx, qy = self.data.qpos[4], self.data.qpos[5]
        up = 1.0 - 2.0 * float(qx*qx + qy*qy)
        return bool(z < MIN_Z or up < 0.5)

    # ----------------------------------------------------------- reset / step
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._stand_qpos.copy()
        self.data.qpos[N_FJ_Q:] += self.np_random.uniform(-0.03, 0.03, size=N_JOINTS)
        mujoco.mj_forward(self.model, self.data)
        self._step  = 0
        self._phase = float(self.np_random.uniform(0, 1))
        self._last_action  = np.zeros(N_JOINTS)
        self._prev_foot_xy = self._foot_xy_z()[0]
        if options and "command" in options:
            self._cmd = np.array(options["command"], dtype=np.float64)
        else:
            self._cmd = self._sample_command()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = DEFAULT_POSE + ACTION_SCALE * action
        # PD position control: target angles → torques, every sim substep
        for _ in range(self._sim_steps):
            q  = self.data.qpos[N_FJ_Q:]
            qd = self.data.qvel[N_FJ_V:]
            tau = np.clip(KP * (target - q) - KD * qd, -TORQUE_LIMIT, TORQUE_LIMIT)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        speed_cmd = max(abs(float(self._cmd[0])), 0.5 * abs(float(self._cmd[1])))
        freq = self._freq_min + (self._freq_max - self._freq_min) * min(speed_cmd / 0.7, 1.0)
        self._phase = (self._phase + self._ctrl_dt * freq) % 1.0

        reward     = self._compute_reward(action)
        self._last_action = action.copy()
        obs        = self._get_obs()
        terminated = self._is_terminated()
        self._step += 1
        truncated  = self._step >= self._max_steps
        info = {
            "x_vel": float(self.data.qvel[0]), "yaw_rate": float(self.data.qvel[5]),
            "torso_z": float(self.data.qpos[2]),
            "cmd_vx": float(self._cmd[0]), "cmd_yaw": float(self._cmd[1]),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
