"""
Environment for the 12-joint robot (robot_v2.xml), with PD POSITION CONTROL.

Control: the policy outputs TARGET joint-angle offsets from the standing pose;
a PD loop converts them to motor torques. Raw torque control (the first
version) let the policy slam actuators to their limits — the body collapsed and
the feet slid while still tracking commanded velocity.

Observations and rewards are BODY-FRAME (projected gravity + body-frame
velocities), so "forward" means "whichever way the robot faces". A world-frame
formulation only teaches the robot to walk in the direction it trained facing.

Obs (50) = [torso_z(1), projected_gravity(3), body_linvel(3), body_angvel(3),
            (jnt_pos - default)(12), jnt_vel(12), last_action(12),
            gait_clock(2), command(2)]
Action (12) = target joint-angle offsets (scaled). Command = [vx, yaw].

Domain randomization (friction / payload / motor gains) and Gaussian sensor
noise are applied when enabled in the config — a policy trained on exact,
noiseless dynamics overfits to them.
"""

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from sim.config import (
    N_JOINTS, N_FJ_Q, N_FJ_V, DEFAULT_POSE, TORQUE_LIMIT,
    KP, KD, ACTION_SCALE, CTRL_DT, STAND_HEIGHT, MIN_Z, UPRIGHT_MIN,
    GAIT_FREQ_MIN, GAIT_FREQ_MAX, GAIT_FREQ_REF, TROT_PAIR_A,
    VX_RANGE, YAW_RANGE, OBS_DIM, EnvCfg,
)

XML_PATH = Path(__file__).parent.parent / "assets" / "robot_v2.xml"
FOOT_NAMES = ["FL", "FR", "RL", "RR"]


class RobotV2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, ctrl_dt=CTRL_DT, cfg: EnvCfg = None):
        super().__init__()
        self.cfg = cfg or EnvCfg()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data  = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer   = None
        self._ctrl_dt    = ctrl_dt
        self._sim_steps  = max(1, round(ctrl_dt / self.model.opt.timestep))

        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self._stand_qpos = self.model.key_qpos[key].copy()

        self._foot_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot")
                           for l in FOOT_NAMES]
        self._foot_r  = float(self.model.geom_size[self._foot_gids[0]][0])
        self._group_A = TROT_PAIR_A

        # Nominal dynamics — randomization is always applied relative to these
        # so repeated resets never compound.
        self._torso_bid       = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._nominal_mass    = float(self.model.body_mass[self._torso_bid])
        self._nominal_fric    = self.model.geom_friction[self._foot_gids, 0].copy()
        # Per-episode actuator params (set by _randomize_dynamics)
        self._kp, self._kd = KP, KD
        self._torque_limit = TORQUE_LIMIT.copy()

        self._phase = 0.0
        self._cmd   = np.zeros(2)
        self._last_action  = np.zeros(N_JOINTS)
        self._prev_foot_xy = np.zeros((4, 2))

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)

        self._step = 0
        self._max_steps = self.cfg.max_steps

    # ----------------------------------------------------------- command API
    def set_command(self, vx, yaw=0.0):
        self._cmd = np.array([vx, yaw], dtype=np.float64)

    def _sample_command(self):
        r = self.np_random.random()
        if r < self.cfg.stand_prob:
            return np.zeros(2)
        elif r < 0.55:
            return np.array([self.np_random.uniform(0.2, VX_RANGE[1]), 0.0])
        elif r < 0.80:
            return np.array([0.0, self.np_random.uniform(*YAW_RANGE)])
        else:
            return np.array([self.np_random.uniform(0.2, 0.5),
                             self.np_random.uniform(-0.6, 0.6)])

    # ------------------------------------------------------ domain randomize
    def set_curriculum(self, scale):
        """Set randomization/noise strength in [0, 1]. Called from the training
        callback so randomization ramps up instead of being full-strength from
        step 0 (which costs nominal-condition performance)."""
        self.cfg.domain_rand.scale = float(scale)
        self.cfg.obs_noise.scale = float(scale)
        return float(scale)

    def _randomize_dynamics(self):
        """Resample physical parameters for this episode."""
        dr = self.cfg.domain_rand
        if not dr.enabled:
            self._kp, self._kd = KP, KD
            self._torque_limit = TORQUE_LIMIT.copy()
            self.model.body_mass[self._torso_bid] = self._nominal_mass
            self.model.geom_friction[self._foot_gids, 0] = self._nominal_fric
            return
        u = self.np_random.uniform
        self.model.geom_friction[self._foot_gids, 0] = (
            self._nominal_fric * u(*dr.range_for("friction")))
        self.model.body_mass[self._torso_bid] = (
            self._nominal_mass + u(*dr.range_for("torso_mass_kg")))
        self._kp = KP * u(*dr.range_for("kp_scale"))
        self._kd = KD * u(*dr.range_for("kd_scale"))
        self._torque_limit = TORQUE_LIMIT * u(*dr.range_for("torque_scale"))

    # ----------------------------------------------------------- helpers
    def _foot_xy_z(self):
        xy = np.array([self.data.geom_xpos[g][:2] for g in self._foot_gids])
        z  = np.array([self.data.geom_xpos[g][2] - self._foot_r for g in self._foot_gids])
        return xy, z

    def _body_frame(self):
        """Body-frame linvel, angvel, and projected gravity (heading-invariant)."""
        quat = self.data.qpos[3:7]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
        lin  = R.T @ self.data.qvel[0:3]
        ang  = R.T @ self.data.qvel[3:6]
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return lin, ang, grav

    def _get_obs(self):
        """Policy observation. Sensor noise is applied HERE only — rewards and
        termination always use the true state, so the metrics stay honest."""
        qpos, qvel = self.data.qpos, self.data.qvel
        lin, ang, grav = self._body_frame()
        clock = np.array([np.sin(2*np.pi*self._phase), np.cos(2*np.pi*self._phase)])

        height    = qpos[2:3].copy()
        joint_pos = qpos[N_FJ_Q:] - DEFAULT_POSE
        joint_vel = qvel[N_FJ_V:].copy()

        n = self.cfg.obs_noise
        s = max(0.0, min(1.0, getattr(n, "scale", 1.0)))
        if n.enabled and s > 0.0:
            g = self.np_random.normal
            height    = height    + g(0, n.height    * s, size=height.shape)
            grav      = grav      + g(0, n.gravity   * s, size=grav.shape)
            lin       = lin       + g(0, n.lin_vel   * s, size=lin.shape)
            ang       = ang       + g(0, n.ang_vel   * s, size=ang.shape)
            joint_pos = joint_pos + g(0, n.joint_pos * s, size=joint_pos.shape)
            joint_vel = joint_vel + g(0, n.joint_vel * s, size=joint_vel.shape)

        return np.concatenate([
            height, grav, lin, ang,
            joint_pos, joint_vel, self._last_action, clock, self._cmd,
        ]).astype(np.float32)

    # ----------------------------------------------------------- reward
    def _compute_reward(self, action):
        w = self.cfg.reward
        qpos, qvel = self.data.qpos, self.data.qvel
        lin, ang, grav = self._body_frame()
        vx, vy   = float(lin[0]), float(lin[1])
        yaw_rate = float(ang[2])
        z        = qpos[2]
        up       = float(-grav[2])

        cmd_vx, cmd_yaw = float(self._cmd[0]), float(self._cmd[1])
        moving = (abs(cmd_vx) > 0.1) or (abs(cmd_yaw) > 0.1)

        # Linear directional progress (dense gradient from a standstill, so the
        # policy can't sit still collecting partial tracking credit) plus an
        # exponential precision bonus that locks onto the exact target.
        prog_vx  = min(vx * np.sign(cmd_vx), abs(cmd_vx)) if abs(cmd_vx) > 0.1 else 0.0
        prog_yaw = min(yaw_rate * np.sign(cmd_yaw), abs(cmd_yaw)) if abs(cmd_yaw) > 0.1 else 0.0
        lin_track = np.exp(-4.0 * (vx - cmd_vx) ** 2)
        yaw_trk   = np.exp(-4.0 * (yaw_rate - cmd_yaw) ** 2)
        track = (w.track_lin_progress * prog_vx + w.track_lin_exp * lin_track
                 + w.track_ang_progress * prog_yaw + w.track_ang_exp * yaw_trk)

        height_reward  = w.height * np.exp(-((z - STAND_HEIGHT) / 0.06) ** 2)
        upright_reward = w.upright * max(0.0, up)
        alive          = w.alive

        foot_xy, foot_z = self._foot_xy_z()
        contacts = foot_z < 0.025
        slip = 0.0
        for i in range(4):
            if contacts[i]:
                slip += np.linalg.norm(foot_xy[i] - self._prev_foot_xy[i]) / self._ctrl_dt
        self._prev_foot_xy = foot_xy
        slip_penalty = w.foot_slip * slip

        if moving:
            A_stance = self._phase < 0.5
            gait = 0.0
            swing_clear = 0.0
            for i in range(4):
                should_stance = (A_stance if i in self._group_A else not A_stance)
                gait += w.gait_match if (bool(contacts[i]) == should_stance) else -w.gait_match
                if not should_stance:
                    swing_clear += min(max(foot_z[i], 0.0), w.target_swing_clear)
            gait += w.swing_clearance * swing_clear
        else:
            gait = 0.5 * (int(contacts.sum()) - 2)

        lateral_penalty   = w.lateral * abs(vy) * (1.0 - min(1.0, abs(cmd_yaw)/0.3))
        yaw_drift_penalty = w.yaw_drift * abs(yaw_rate) if abs(cmd_yaw) < 0.1 else 0.0
        stillness_penalty = (w.stillness * (abs(vx) + abs(vy) + abs(yaw_rate))
                             if not moving else 0.0)
        roll_pitch_pen    = w.roll_pitch * float(ang[0]**2 + ang[1]**2)
        action_rate_pen   = w.action_rate * float(np.sum((action - self._last_action) ** 2))
        jvel_pen          = w.joint_vel * float(np.sum(qvel[N_FJ_V:] ** 2))
        foot_lift_pen     = w.foot_lift * sum(max(0.0, fz - w.max_foot_lift) for fz in foot_z)

        return (track + height_reward + upright_reward + alive + gait
                - slip_penalty - lateral_penalty - yaw_drift_penalty
                - stillness_penalty - roll_pitch_pen - action_rate_pen
                - jvel_pen - foot_lift_pen)

    def _is_terminated(self):
        _, _, grav = self._body_frame()
        return bool(self.data.qpos[2] < MIN_Z or float(-grav[2]) < UPRIGHT_MIN)

    # ----------------------------------------------------------- reset / step
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_dynamics()
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
        for _ in range(self._sim_steps):
            q  = self.data.qpos[N_FJ_Q:]
            qd = self.data.qvel[N_FJ_V:]
            tau = np.clip(self._kp * (target - q) - self._kd * qd,
                          -self._torque_limit, self._torque_limit)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        speed_cmd = max(abs(float(self._cmd[0])), 0.5 * abs(float(self._cmd[1])))
        freq = GAIT_FREQ_MIN + (GAIT_FREQ_MAX - GAIT_FREQ_MIN) * min(speed_cmd / GAIT_FREQ_REF, 1.0)
        self._phase = (self._phase + self._ctrl_dt * freq) % 1.0

        reward = self._compute_reward(action)
        self._last_action = action.copy()
        obs = self._get_obs()
        terminated = self._is_terminated()
        self._step += 1
        truncated = self._step >= self._max_steps

        lin, ang, _ = self._body_frame()
        info = {
            "x_vel": float(lin[0]), "yaw_rate": float(ang[2]),   # body frame
            "torso_z": float(self.data.qpos[2]),
            "cmd_vx": float(self._cmd[0]), "cmd_yaw": float(self._cmd[1]),
        }
        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------- render
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
