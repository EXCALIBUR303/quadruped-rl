"""
Stage E — TALK TO YOUR ROBOT (12-joint robot_v2).

Speak a command ("walk forward", "turn left", "go faster", "stop") and the robot
obeys in the MuJoCo viewer and confirms out loud. Say "quit" to exit.

    mjpython brain/talk_to_robot.py
Requires the Ollama server:  ollama serve  (or `brew services start ollama`)
"""

import sys, os, threading
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from controller import RobotController
import nlu
import voice

console = Console()
QUIT_WORDS = ("quit", "exit", "goodbye", "shut down", "go to sleep")


def banner():
    art = Text()
    art.append("  ▄▄▄  \n", style="bold cyan")
    art.append(" ╱   ╲  ", style="cyan"); art.append("R O B O - P U P\n", style="bold white")
    art.append(" ╲▂▂▂╱  ", style="cyan"); art.append("voice-controlled quadruped\n", style="dim")
    console.print(Panel(art, border_style="cyan", title="[bold]🐕 online[/bold]",
                        subtitle="say 'walk', 'turn left', 'faster', 'stop', 'quit'"))


def status(heard, cmd):
    t = Text()
    t.append("🎤 ", style="bold")
    t.append(f"{heard}\n", style="italic white")
    t.append("   → ", style="dim")
    t.append(f"speed {cmd['vx']:+.2f}  turn {cmd['yaw']:+.2f}", style="bold cyan")
    t.append(f"   💬 {cmd['reply']}", style="green")
    console.print(t)


def voice_loop(ctrl):
    voice.speak("Robo pup online. Tell me to walk, turn, or stop.")
    while True:
        console.print("[dim]listening…[/dim]", end="\r")
        text = voice.listen_until_silence(verbose=False)
        if not text:
            continue
        if any(w in text.lower() for w in QUIT_WORDS):
            console.print("[bold red]powering down 🔌[/bold red]")
            voice.speak("Powering down. Goodbye!")
            os._exit(0)
        cmd = nlu.parse_command(text, current_vx=ctrl.speed, current_yaw=ctrl.turn)
        ctrl.set_command(cmd["vx"], cmd["yaw"])
        status(text, cmd)
        voice.speak(cmd["reply"])


if __name__ == "__main__":
    console.print("[dim]loading policy + speech model…[/dim]")
    ctrl = RobotController()
    voice._get_model()
    banner()
    threading.Thread(target=lambda: voice_loop(ctrl), daemon=True).start()
    ctrl.run_viewer()
