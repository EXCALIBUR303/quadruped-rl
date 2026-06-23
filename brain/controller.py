"""
Stage B — RobotController: loads the trained speed-control policy and runs it
in the MuJoCo viewer in real time, exposing a live set_command(vx) that the
voice layer (or the keyboard) drives.

Standalone keyboard demo (macOS needs mjpython for the viewer):
    mjpython brain/controller.py
Then type a speed 0.0-0.7, 's' to stop, 'q' to quit.
"""

import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.envs import QuadrupedEnv

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


class RobotController:
    """Runs the trained policy; command can be changed live and safely."""

    def __init__(self, model_path=None, norm_path=None):
        model_path = model_path or os.path.join(CKPT, "best_model")
        norm_path  = norm_path  or os.path.join(CKPT, "vec_normalize.pkl")

        self.raw = QuadrupedEnv(render_mode="human")
        venv = DummyVecEnv([lambda: self.raw])
        self.venv = VecNormalize.load(norm_path, venv)
        self.venv.training = False
        self.venv.norm_reward = False
        self.model = PPO.load(model_path, env=self.venv)

        self.obs  = self.venv.reset()
        self._cmd = [0.0, 0.0]
        self.raw.set_command(0.0, 0.0)

    # ---- live command interface (thread-safe enough for our use) ----------
    def set_command(self, vx, yaw=0.0):
        self._cmd = [float(vx), float(yaw)]

    @property
    def speed(self):
        return self._cmd[0]

    # ---- simulation stepping ---------------------------------------------
    def step(self):
        self.raw.set_command(self._cmd[0], self._cmd[1])
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, _, done, _ = self.venv.step(action)
        if done[0]:
            self.obs = self.venv.reset()
            self.raw.set_command(self._cmd[0], self._cmd[1])

    def run_viewer(self, hz=50, on_start=None):
        """Real-time viewer loop (call from the main thread on macOS)."""
        dt = 1.0 / hz
        with mujoco.viewer.launch_passive(self.raw.model, self.raw.data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = self.raw.model.body("torso").id
            v.cam.distance = 1.5
            if on_start:
                on_start()
            while v.is_running():
                t0 = time.time()
                self.step()
                v.sync()
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)


if __name__ == "__main__":
    ctrl = RobotController()

    def keyboard_loop():
        print("\n--- keyboard control ---")
        print("type a speed 0.0-0.7 | 's' = stop | 'q' = quit, then Enter:")
        while True:
            try:
                cmd = input("> ").strip().lower()
            except EOFError:
                return
            if cmd == "q":
                os._exit(0)
            elif cmd == "s":
                ctrl.set_command(0.0); print("  stop")
            else:
                try:
                    vx = max(0.0, min(0.7, float(cmd)))
                    ctrl.set_command(vx); print(f"  vx = {vx:.2f}")
                except ValueError:
                    print("  enter a number 0.0-0.7, 's', or 'q'")

    threading.Thread(target=keyboard_loop, daemon=True).start()
    ctrl.run_viewer()
