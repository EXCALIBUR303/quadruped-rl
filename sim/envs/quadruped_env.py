"""
Command-conditioned Gymnasium environment for the 8-servo quadruped.

The policy is given a velocity COMMAND each episode and rewarded for tracking
it, so the same network can stand, walk at any speed, and turn — which is what
the voice/LLM layer drives later.

Obs (31) = [torso_z(1), quat(4), linvel(3), angvel(3),
            jnt_pos(8), jnt_vel(8), gait_clock(2), command(2)]
Action (8) = normalised motor commands matching real servos 1-to-1.
Command  = [cmd_vx (m/s forward), cmd_yaw (rad/s turn)]

A periodic gait clock drives a diagonal trot (FL+RR / FR+RL) when moving and
relaxes to a 4-foot stand when commanded to stop.
"""

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH   = Path(__file__).parent.parent / "assets" / "quadruped.xml"
N_JOINTS   = 8
N_FJ_Q     = 7
N_FJ_V     = 6
STAND_Z    = 0.24
MIN_Z      = 0.10

# Command sampling ranges (kept within what this 8-servo robot can hit stably)
VX_RANGE   = (0.0, 0.7)    # m/s forward (no backward — unreliable on this body)
YAW_RANGE  = (-0.5, 0.5)   # rad/s turn rate (realised as curved walking)
STAND_PROB = 0.15          # fraction of episodes commanded to stand still


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, ctrl_dt=0.02):
        super().__init__()
        self.model    = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data     = mujoco.MjData(self.model)
        self.render_mode  = render_mode
        self._renderer    = None
        self._ctrl_dt     = ctrl_dt
        self._sim_steps   = max(1, round(ctrl_dt / self.model.opt.timestep))

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self._stand_qpos = self.model.key_qpos[key_id].copy()

        self._foot_gids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
            for leg in ["FL", "FR", "RL", "RR"]
        ]
        self._foot_r        = float(self.model.geom_size[self._foot_gids[0]][0])
        self._max_foot_lift = 0.10
        self._group_A       = (0, 3)        # FL + RR

        # Gait cadence scales with commanded speed: step faster to go faster
        # (a fixed cadence caps top speed).  freq in Hz, period = 1/freq.
        self._freq_min    = 2.2       # Hz at a standstill / slow
        self._freq_max    = 4.2       # Hz at top commanded speed
        self._phase       = 0.0
        self._cmd         = np.zeros(2, dtype=np.float32)  # [vx, yaw]

        n_obs = 1 + 4 + 3 + 3 + N_JOINTS + N_JOINTS + 2 + 2  # 31
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)

        self._step        = 0
        self._max_steps   = 1000
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float32)

    # ------------------------------------------------------------ command API
    def set_command(self, vx, yaw):
        """Live command setter — used by the control API / voice layer."""
        self._cmd = np.array([vx, yaw], dtype=np.float32)

    def _sample_command(self):
        # Speed control only: stand still, or walk forward at a sampled speed.
        # Turning was dropped — this 8-servo body (no hip abduction) cannot
        # turn without destabilising the gait; documented hardware limitation.
        if self.np_random.random() < 0.15:             # stand still
            return np.zeros(2, dtype=np.float32)
        return np.array([self.np_random.uniform(0.1, 0.7), 0.0], np.float32)

    # ------------------------------------------------------------------ obs
    def _get_obs(self):
        qpos, qvel = self.data.qpos, self.data.qvel
        clock = np.array([np.sin(2 * np.pi * self._phase),
                          np.cos(2 * np.pi * self._phase)])
        return np.concatenate([
            qpos[2:3], qpos[3:7], qvel[0:3], qvel[3:6],
            qpos[N_FJ_Q:], qvel[N_FJ_V:], clock, self._cmd,
        ]).astype(np.float32)

    def _foot_contacts(self):
        return np.array([
            (float(self.data.geom_xpos[g][2]) - self._foot_r) < 0.025
            for g in self._foot_gids
        ])

    # ------------------------------------------------------------------ reward
    def _compute_reward(self, action):
        qpos, qvel = self.data.qpos, self.data.qvel
        vx, vy   = float(qvel[0]), float(qvel[1])
        yaw_rate = float(qvel[5])
        z        = qpos[2]
        qx, qy   = qpos[4], qpos[5]
        up = 1.0 - 2.0 * float(qx * qx + qy * qy)

        cmd_vx, cmd_yaw = float(self._cmd[0]), float(self._cmd[1])
        moving = (abs(cmd_vx) > 0.1) or (abs(cmd_yaw) > 0.1)

        # ---- Command tracking (the core objective) ------------------------
        # Linear directional-progress term pays for velocity TOWARD the command
        # (capped at the target, zero when standing) → gradient from a standstill
        # and no "do nothing" trap.  An exp precision bonus locks onto the target.
        prog_vx  = min(vx * np.sign(cmd_vx), abs(cmd_vx)) if abs(cmd_vx) > 0.1 else 0.0
        prog_yaw = min(yaw_rate * np.sign(cmd_yaw), abs(cmd_yaw)) if abs(cmd_yaw) > 0.1 else 0.0
        lin_track = np.exp(-4.0 * (vx - cmd_vx) ** 2)
        yaw_track = np.exp(-4.0 * (yaw_rate - cmd_yaw) ** 2)
        track_reward = (5.0 * prog_vx + 1.5 * lin_track
                        + 2.5 * prog_yaw + 1.0 * yaw_track)

        # Free posture rewards kept small so standing is not its own attractor.
        upright_reward = 0.5 * max(0.0, up)
        height_reward  = 0.5 * max(0.0, 1.0 - abs(z - STAND_Z) / STAND_Z)
        alive          = 0.3

        # ---- Gait reward: trot when moving, stand when stopped ------------
        contacts = self._foot_contacts()
        if moving:
            A_stance = self._phase < 0.5
            gait_reward = 0.0
            for i in range(4):
                should_stance = (A_stance if i in self._group_A else not A_stance)
                correct = bool(contacts[i]) == should_stance
                gait_reward += 0.4 if correct else -0.4
        else:
            # standing: reward all four feet planted
            gait_reward = 0.5 * (int(contacts.sum()) - 2)   # +1 at 4 feet, -1 at 0

        # ---- Penalties ----------------------------------------------------
        # Lateral drift only penalised when not turning (turning needs some vy).
        lateral_penalty    = 0.4 * abs(vy) * (1.0 - min(1.0, abs(cmd_yaw) / 0.3))
        # Unwanted yaw drift when going straight → keeps the line straight.
        yaw_drift_penalty  = 1.2 * abs(yaw_rate) if abs(cmd_yaw) < 0.1 else 0.0
        # Stillness: when commanded to stand, punish ALL motion hard so the
        # policy can't drift/spin in place (the earlier instability).
        stillness_penalty  = (1.5 * (abs(vx) + abs(vy) + abs(yaw_rate))
                              if not moving else 0.0)
        roll_pitch_penalty = 0.3 * float(qvel[3] ** 2 + qvel[4] ** 2)
        smooth_penalty     = 0.005 * float(np.sum((action - self._prev_action) ** 2))
        energy_penalty     = 0.001 * float(np.sum(action ** 2))
        joint_vel_penalty  = 0.0001 * float(np.sum(qvel[N_FJ_V:] ** 2))

        foot_lift_penalty = 0.0
        for g in self._foot_gids:
            fz = float(self.data.geom_xpos[g][2]) - self._foot_r
            foot_lift_penalty += max(0.0, fz - self._max_foot_lift)
        foot_lift_penalty *= 1.5

        return (track_reward + upright_reward + height_reward + alive + gait_reward
                - lateral_penalty - yaw_drift_penalty - stillness_penalty
                - roll_pitch_penalty - smooth_penalty - energy_penalty
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
        self._phase       = float(self.np_random.uniform(0, 1))
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float32)

        # Command: explicit override via options, else random
        if options and "command" in options:
            self._cmd = np.array(options["command"], dtype=np.float32)
        else:
            self._cmd = self._sample_command()
        return self._get_obs(), {}

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        for _ in range(self._sim_steps):
            mujoco.mj_step(self.model, self.data)
        # advance gait clock at a speed-dependent cadence
        speed_cmd = abs(float(self._cmd[0]))
        freq = self._freq_min + (self._freq_max - self._freq_min) * min(speed_cmd / 0.7, 1.0)
        self._phase = (self._phase + self._ctrl_dt * freq) % 1.0

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
            "yaw_rate": float(self.data.qvel[5]),
            "cmd_vx":  float(self._cmd[0]),
            "cmd_yaw": float(self._cmd[1]),
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
