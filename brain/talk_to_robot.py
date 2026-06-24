"""
Stage E — TALK TO YOUR ROBOT.

Speak a command ("walk forward", "go faster", "slow down", "stop") and the
robot obeys in the MuJoCo viewer and confirms out loud.  Say "quit" to exit.

Run (macOS needs mjpython for the viewer):
    mjpython brain/talk_to_robot.py

Requires the Ollama server running:  ollama serve  (in another terminal)
"""

import sys, os, threading
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from controller import RobotController
import nlu
import voice

QUIT_WORDS = ("quit", "exit", "goodbye", "shut down", "go to sleep")


def voice_loop(ctrl):
    voice.speak("Hi! I'm ready. Tell me to walk, speed up, slow down, or stop.")
    while True:
        text = voice.listen_until_silence()
        if not text:
            continue
        if any(w in text.lower() for w in QUIT_WORDS):
            voice.speak("Goodbye!")
            os._exit(0)
        cmd = nlu.parse_command(text, current_vx=ctrl.speed, current_yaw=ctrl.turn)
        ctrl.set_command(cmd["vx"], cmd["yaw"])
        print(f"  -> vx={cmd['vx']:+.2f} yaw={cmd['yaw']:+.2f}  | {cmd['reply']}")
        voice.speak(cmd["reply"])


if __name__ == "__main__":
    print("Loading walking policy ...")
    ctrl = RobotController()
    print("Loading speech model ...")
    voice._get_model()                       # preload Whisper so first cmd is fast
    threading.Thread(target=lambda: voice_loop(ctrl), daemon=True).start()
    print("Viewer opening — talk to your robot!")
    ctrl.run_viewer()
