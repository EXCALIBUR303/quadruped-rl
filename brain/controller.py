"""
RobotController — mode-aware brain-to-body bridge for the 12-joint robot.

Modes:
  POLICY   : the trained RL policy walks / turns per set_command(vx, yaw)
  TRICK    : choreographed keyframe motion (sit, wave, dance, bow, lie, wiggle)
             played through the PD controller directly
  NAVIGATE : fetch mode — auto-steers the walking policy toward the glowing
             ball, celebrates on arrival
  SPIN     : timed turn-in-place composite

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

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints_v2")
KP, KD = 25.0, 0.7
TORQUE_LIMIT = np.array([8.0, 12.0, 14.0] * 4)

# state → (eye rgba, emission)
EYE_STATES = {
    "idle":     ([0.30, 1.00, 1.00, 1.0], 0.95),   # cyan
    "moving":   ([0.30, 1.00, 1.00, 1.0], 0.95),
    "trick":    ([1.00, 0.60, 0.10, 1.0], 0.95),   # orange
    "navigate": ([0.30, 1.00, 0.45, 1.0], 0.95),   # green
}


class RobotController:
    def __init__(self, model_path=None, norm_path=None):
        model_path = model_path or os.path.join(CKPT, "best_model")
        norm_path  = norm_path  or os.path.join(CKPT, "vec_normalize.pkl")

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
        self._spin_left  = 0
        self.on_event    = None              # callback(str) for arrival messages

        # fetch ball (mocap body)
        bid = mujoco.mj_name2id(self.raw.model, mujoco.mjtObj.mjOBJ_BODY, "fetch_ball")
        self._ball_mocap = self.raw.model.body_mocapid[bid]
        self._eye_mat    = mujoco.mj_name2id(self.raw.model, mujoco.mjtObj.mjOBJ_MATERIAL, "eye")
        self._set_eyes("idle")

    # ------------------------------------------------------------ appearance
    def _set_eyes(self, state):
        rgba, em = EYE_STATES.get(state, EYE_STATES["idle"])
        self.raw.model.mat_rgba[self._eye_mat]     = rgba
        self.raw.model.mat_emission[self._eye_mat] = em

    # ------------------------------------------------------------ commands
    def set_command(self, vx, yaw=0.0):
        """Walk/turn command — also cancels any trick/navigation."""
        self._cmd = [float(vx), float(yaw)]
        self.mode = "POLICY"
        self._set_eyes("moving" if (abs(vx) + abs(yaw)) > 0.05 else "idle")

    def do_trick(self, name):
        if name not in tricks.TRICKS:
            return False
        self._trick_seq = tricks.compile_trick(name, start_pose=self.raw.data.qpos[7:].copy())
        self._trick_i   = 0
        self.mode       = "TRICK"
        self._set_eyes("trick")
        return True

    def spin(self, seconds=6.0, rate=0.65):
        self._cmd = [0.0, rate]
        self._spin_left = int(seconds * 50)
        self.mode = "SPIN"
        self._set_eyes("trick")

    def fetch(self):
        """Drop the ball somewhere ahead-ish and go get it."""
        yaw = self._yaw()
        ang  = yaw + np.random.uniform(-1.2, 1.2)
        dist = np.random.uniform(1.6, 2.6)
        pos  = self.raw.data.qpos[:2] + dist * np.array([np.cos(ang), np.sin(ang)])
        self.raw.data.mocap_pos[self._ball_mocap] = [pos[0], pos[1], 0.05]
        self.mode = "NAVIGATE"
        self._set_eyes("navigate")

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

    # ------------------------------------------------------------ main step (50 Hz)
    def step(self):
        if self.mode == "TRICK":
            target = self._trick_seq[self._trick_i]
            for _ in range(4):                        # 4 x 0.005 s substeps
                q, qd = self.raw.data.qpos[7:], self.raw.data.qvel[6:]
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
            pos  = self.raw.data.qpos[:2]
            delta = ball - pos
            dist  = float(np.linalg.norm(delta))
            if dist < 0.32:                           # arrived!
                self.raw.data.mocap_pos[self._ball_mocap] = [0, 0, -1]   # "picked up"
                self.do_trick("wiggle")
                if self.on_event:
                    self.on_event("fetched")
                return
            err = np.arctan2(delta[1], delta[0]) - self._yaw()
            err = (err + np.pi) % (2 * np.pi) - np.pi
            yaw_cmd = float(np.clip(1.4 * err, -0.7, 0.7))
            vx_cmd  = 0.45 if abs(err) < 0.7 else 0.12
            self._policy_step(vx_cmd, yaw_cmd)

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
        print("t sit|wave|dance|bow|lie  |  f=fetch  |  p=spin")
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
