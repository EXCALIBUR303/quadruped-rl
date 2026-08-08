"""
RobotController — mode-aware brain-to-body bridge for the 12-joint robot.

Modes:
  POLICY   : the trained RL policy walks / turns per set_command(vx, yaw)
  TRICK    : choreographed keyframe motion (sit, wave, dance, bow, lie, wiggle)
             played through the PD controller directly
  NAVIGATE : fetch mode — auto-steers the walking policy toward the glowing
             ball, steering around the crates, celebrates on arrival
  PATROL   : loops a circuit of waypoints (glowing flag marks the current one)
  FOLLOW   : the ball wanders around the arena and the robot tails it
  SPIN     : timed turn-in-place composite

All autonomous modes share one obstacle-aware steering law (potential field:
attraction to the target + repulsion from the crates).

Extras:
  - Eye + accent colors reflect state (idle cyan / trick orange / fetch green)
  - Fixes the "20-second teleport" glitch: the env's episode truncation used to
    auto-reset (teleport) the robot mid-demo; episodes are now unbounded here.

Keyboard demo (macOS needs mjpython):
    mjpython brain/controller.py
  number = speed | l/r = turn | s = stop | q = quit
  t sit|wave|dance|bow|lie  = trick     | f = fetch the ball
"""

import sys, os, time, threading
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.envs import RobotV2Env
import tricks
import preflight

from sim.config import KP, KD, TORQUE_LIMIT, DEFAULT_POSE, CTRL_HZ, N_FJ_Q, N_FJ_V

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints_v2")

# state → (eye rgba, emission)
EYE_STATES = {
    "idle":     ([0.30, 1.00, 1.00, 1.0], 0.95),   # cyan
    "moving":   ([0.30, 1.00, 1.00, 1.0], 0.95),
    "trick":    ([1.00, 0.60, 0.10, 1.0], 0.95),   # orange
    "navigate": ([0.30, 1.00, 0.45, 1.0], 0.95),   # green
}


class RobotController:
    def __init__(self, model_path=None, norm_path=None):
        if model_path is None or norm_path is None:
            # Raises PreflightError with the exact command to fix it, instead
            # of a bare FileNotFoundError from deep inside SB3.
            found_model, found_norm = preflight.check_policy(CKPT)
            model_path = model_path or found_model
            norm_path  = norm_path  or found_norm

        self.raw = RobotV2Env(render_mode="human")
        self.raw._max_steps = 10 ** 9        # kill the 20 s truncation-teleport
        venv = DummyVecEnv([lambda: self.raw])
        self.venv = VecNormalize.load(norm_path, venv)
        self.venv.training = False
        self.venv.norm_reward = False
        self.model = PPO.load(model_path, env=self.venv)

        self.obs   = self.venv.reset()
        self._cmd  = [0.0, 0.0]
        self.raw.set_command(0.0, 0.0)

        self.mode        = "POLICY"
        self._trick_seq  = None
        self._trick_i    = 0
        # The voice/camera thread mutates mode state while the viewer thread is
        # inside step(); without this, swapping _trick_seq between step()'s read
        # of the sequence and its index raises IndexError. Re-entrant because
        # step() itself calls do_trick() for the fetch celebration.
        self._lock = threading.RLock()
        self._spin_left  = 0
        self.on_event    = None              # callback(str) for arrival messages

        # fetch ball + patrol flag (mocap bodies)
        m = self.raw.model
        self._ball_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "fetch_ball")]
        self._flag_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "waypoint_flag")]
        self._eye_mat    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "eye")

        # obstacle crates: read positions/sizes from the model by name
        self._obstacles = []
        for i in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.startswith("crate"):
                x, y = m.geom_pos[i][:2]
                r = float(np.hypot(m.geom_size[i][0], m.geom_size[i][1]))
                self._obstacles.append((float(x), float(y), r))

        # patrol circuit (kept clear of the crates; avoidance covers the rest)
        self._waypoints = [(2.6, 1.6), (3.4, -0.6), (1.6, -1.9), (0.3, 0.2)]
        self._wp_i = 0
        # follow-mode ball velocity (random walk)
        self._ball_vel = np.zeros(2)

        self._set_eyes("idle")

    # ------------------------------------------------------------ appearance
    def _set_eyes(self, state):
        rgba, em = EYE_STATES.get(state, EYE_STATES["idle"])
        self.raw.model.mat_rgba[self._eye_mat]     = rgba
        self.raw.model.mat_emission[self._eye_mat] = em

    # ------------------------------------------------------------ commands
    def set_command(self, vx, yaw=0.0):
        """Walk/turn command — also cancels any trick/autonomous mode."""
        with self._lock:
            self._cmd = [float(vx), float(yaw)]
            self.mode = "POLICY"
            self._hide_flag()
            self.raw.data.mocap_pos[self._ball_mocap] = [0, 0, -1]
            self._set_eyes("moving" if (abs(vx) + abs(yaw)) > 0.05 else "idle")

    def do_trick(self, name):
        if name not in tricks.TRICKS:
            return False
        with self._lock:
            self._trick_seq = tricks.compile_trick(
                name, start_pose=self.raw.data.qpos[N_FJ_Q:].copy())
            self._trick_i   = 0
            self.mode       = "TRICK"
            self._set_eyes("trick")
        return True

    def spin(self, seconds=6.0, rate=0.65):
        with self._lock:
            self._cmd = [0.0, rate]
            self._spin_left = int(seconds * CTRL_HZ)
            self.mode = "SPIN"
            self._set_eyes("trick")

    def fetch(self):
        """Drop the ball somewhere ahead-ish and go get it."""
        with self._lock:
            yaw = self._yaw()
            ang  = yaw + np.random.uniform(-1.2, 1.2)
            dist = np.random.uniform(1.6, 2.6)
            pos  = self.raw.data.qpos[:2] + dist * np.array([np.cos(ang), np.sin(ang)])
            self.raw.data.mocap_pos[self._ball_mocap] = [pos[0], pos[1], 0.05]
            self.mode = "NAVIGATE"
            self._set_eyes("navigate")

    def patrol(self):
        """Loop the waypoint circuit until told otherwise."""
        with self._lock:
            self._wp_i = 0
            self._show_flag(self._waypoints[0])
            self.mode = "PATROL"
            self._set_eyes("navigate")

    def follow(self):
        """The ball wanders around the arena; the robot tails it."""
        with self._lock:
            yaw = self._yaw()
            pos = self.raw.data.qpos[:2] + 1.2 * np.array([np.cos(yaw), np.sin(yaw)])
            self.raw.data.mocap_pos[self._ball_mocap] = [pos[0], pos[1], 0.05]
            self._ball_vel = np.zeros(2)
            self.mode = "FOLLOW"
            self._set_eyes("navigate")

    def _show_flag(self, xy):
        self.raw.data.mocap_pos[self._flag_mocap] = [xy[0], xy[1], 0.0]

    def _hide_flag(self):
        self.raw.data.mocap_pos[self._flag_mocap] = [0, 0, -1]

    @property
    def speed(self): return self._cmd[0]
    @property
    def turn(self):  return self._cmd[1]

    # ------------------------------------------------------------ helpers
    def _yaw(self):
        qw, qx, qy, qz = self.raw.data.qpos[3:7]
        return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def _policy_step(self, vx, yaw):
        self.raw.set_command(vx, yaw)
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, _, done, _ = self.venv.step(action)
        if done[0]:                                   # fell — get back up
            self.obs = self.venv.reset()
            self.raw.set_command(vx, yaw)

    def _resync_obs(self):
        """Recompute normalized policy obs after direct (non-venv) stepping."""
        self.obs = self.venv.normalize_obs(self.raw._get_obs()[None, :].astype(np.float32))

    def _steer_toward(self, target_xy, speed=0.45):
        """Obstacle-aware steering (potential field): walk toward the target,
        pushed away from any crate we get close to. Returns distance left."""
        pos   = self.raw.data.qpos[:2]
        delta = np.asarray(target_xy) - pos
        dist  = float(np.linalg.norm(delta))
        att   = delta / (dist + 1e-6)

        rep = np.zeros(2)
        for ox, oy, orad in self._obstacles:
            v = pos - np.array([ox, oy])
            d = float(np.linalg.norm(v))
            influence = orad + 0.45              # start dodging this far out
            if d < influence:
                rep += (v / (d + 1e-6)) * 2.2 * (influence - d) / influence

        head = att + rep
        err  = np.arctan2(head[1], head[0]) - self._yaw()
        err  = (err + np.pi) % (2 * np.pi) - np.pi

        # Keep commands INSIDE the policy's training distribution (pure turn at
        # vx=0, walk+turn with vx>=0.2, or near-straight walking). Smoothly
        # blended intermediate commands are out-of-distribution and make the
        # policy orbit the target instead of reaching it.
        if abs(err) > 0.9:
            vx_cmd, yaw_cmd = 0.0, float(np.sign(err) * 0.6)      # turn in place
        elif abs(err) > 0.35:
            vx_cmd, yaw_cmd = 0.3, float(np.clip(1.0 * err, -0.6, 0.6))
        else:
            vx_cmd = min(speed, 0.3 if dist < 0.7 else speed)      # slow near target
            yaw_cmd = float(np.clip(0.8 * err, -0.4, 0.4))
        self._policy_step(vx_cmd, yaw_cmd)
        return dist

    # ------------------------------------------------------------ main step (50 Hz)
    def step(self):
      # Held for the whole dispatch so a mode switch from the voice/camera
      # thread cannot land between reading the mode and acting on it.
      with self._lock:
        if self.mode == "TRICK":
            target = self._trick_seq[self._trick_i]
            for _ in range(4):                        # 4 x 0.005 s substeps
                q, qd = self.raw.data.qpos[N_FJ_Q:], self.raw.data.qvel[N_FJ_V:]
                tau = np.clip(KP * (target - q) - KD * qd, -TORQUE_LIMIT, TORQUE_LIMIT)
                self.raw.data.ctrl[:] = tau
                mujoco.mj_step(self.raw.model, self.raw.data)
            self._trick_i += 1
            if self._trick_i >= len(self._trick_seq):
                self._trick_seq = None
                self._resync_obs()
                self._cmd = [0.0, 0.0]
                self.mode = "POLICY"
                self._set_eyes("idle")

        elif self.mode == "NAVIGATE":
            ball = self.raw.data.mocap_pos[self._ball_mocap][:2]
            if self._steer_toward(ball) < 0.32:       # arrived!
                self.raw.data.mocap_pos[self._ball_mocap] = [0, 0, -1]   # "picked up"
                self.do_trick("wiggle")
                if self.on_event:
                    self.on_event("fetched")

        elif self.mode == "PATROL":
            wp = self._waypoints[self._wp_i]
            if self._steer_toward(wp) < 0.35:
                self._wp_i = (self._wp_i + 1) % len(self._waypoints)
                self._show_flag(self._waypoints[self._wp_i])
                if self.on_event and self._wp_i == 0:
                    self.on_event("lap")

        elif self.mode == "FOLLOW":
            # ball wanders: smooth random-walk velocity, bounced off arena walls
            self._ball_vel += np.random.uniform(-0.05, 0.05, size=2)
            spd = np.linalg.norm(self._ball_vel)
            if spd > 0.22:                            # slow enough to keep up with
                self._ball_vel *= 0.22 / spd
            ball = self.raw.data.mocap_pos[self._ball_mocap][:2] + self._ball_vel * 0.02
            for k, (lo, hi) in enumerate([(-0.5, 3.8), (-2.5, 2.5)]):
                if not (lo < ball[k] < hi):
                    self._ball_vel[k] *= -1
                    ball[k] = np.clip(ball[k], lo, hi)
            self.raw.data.mocap_pos[self._ball_mocap] = [ball[0], ball[1], 0.05]
            dist = float(np.linalg.norm(ball - self.raw.data.qpos[:2]))
            if dist > 0.55:                            # keep a polite distance
                self._steer_toward(ball, speed=0.55)
            else:
                self._policy_step(0.0, 0.0)

        elif self.mode == "SPIN":
            self._policy_step(0.0, self._cmd[1])
            self._spin_left -= 1
            if self._spin_left <= 0:
                self._cmd = [0.0, 0.0]
                self.mode = "POLICY"
                self._set_eyes("idle")

        else:                                         # POLICY
            self._policy_step(self._cmd[0], self._cmd[1])

    # ------------------------------------------------------------ viewer loop
    def run_viewer(self, hz=50, on_start=None):
        dt = 1.0 / hz
        with mujoco.viewer.launch_passive(self.raw.model, self.raw.data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = self.raw.model.body("torso").id
            v.cam.distance = 1.7
            v.cam.elevation = -16
            if on_start:
                on_start()
            while v.is_running():
                t0 = time.time()
                self.step()
                v.sync()
                el = time.time() - t0
                if el < dt:
                    time.sleep(dt - el)


if __name__ == "__main__":
    ctrl = RobotController()

    def keyboard_loop():
        print("\n--- keyboard control ---")
        print("number=speed | l/r=turn | s=stop | q=quit")
        print("t sit|wave|dance|bow|lie | f=fetch | g=patrol | b=follow | p=spin")
        while True:
            try:
                cmd = input("> ").strip().lower()
            except EOFError:
                return
            if cmd == "q":
                os._exit(0)
            elif cmd == "s":
                ctrl.set_command(0.0, 0.0); print("  stop")
            elif cmd == "l":
                ctrl.set_command(ctrl.speed, 0.6); print("  turn left")
            elif cmd == "r":
                ctrl.set_command(ctrl.speed, -0.6); print("  turn right")
            elif cmd == "f":
                ctrl.fetch(); print("  fetching the ball!")
            elif cmd == "g":
                ctrl.patrol(); print("  patrolling the arena!")
            elif cmd == "b":
                ctrl.follow(); print("  following the ball!")
            elif cmd == "p":
                ctrl.spin(); print("  spinning!")
            elif cmd.startswith("t "):
                name = cmd[2:].strip()
                print(f"  trick: {name}" if ctrl.do_trick(name) else f"  unknown trick {name!r}")
            else:
                try:
                    vx = max(-0.3, min(0.6, float(cmd)))
                    ctrl.set_command(vx, 0.0); print(f"  vx = {vx:.2f}")
                except ValueError:
                    print("  number | l r s q | t <trick> | f | p")

    threading.Thread(target=keyboard_loop, daemon=True).start()
    ctrl.run_viewer()
