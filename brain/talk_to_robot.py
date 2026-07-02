"""
TALK TO YOUR ROBOT — full voice demo for the 12-joint robot pup.

Speak commands and it obeys in the MuJoCo viewer and answers out loud:
  movement : "walk forward", "turn left", "faster", "back up", "stop"
  tricks   : "sit", "lie down", "wave hello", "dance", "take a bow",
             "spin around", "go fetch the ball!"
  chat     : ask it anything — it answers in character

Say "goodbye" to exit.

    mjpython brain/talk_to_robot.py
Requires Ollama running:  brew services start ollama
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
QUIT_WORDS = ("goodbye", "good bye", "power down", "shut down", "quit", "exit")


def banner():
    art = Text()
    art.append("  ▄▄▄  \n", style="bold cyan")
    art.append(" ╱   ╲  ", style="cyan"); art.append("R O B O - P U P  2.0\n", style="bold white")
    art.append(" ╲▂▂▂╱  ", style="cyan"); art.append("walks · turns · sits · waves · dances · fetches\n", style="dim")
    console.print(Panel(art, border_style="cyan", title="[bold]🐕 online[/bold]",
                        subtitle="try: 'sit' · 'dance' · 'fetch the ball' · 'goodbye'"))


def status(heard, intent):
    t = Text()
    t.append("🎤 ", style="bold")
    t.append(f"{heard}\n", style="italic white")
    if intent["action"] == "move":
        t.append("   → ", style="dim")
        t.append(f"speed {intent['vx']:+.2f}  turn {intent['yaw']:+.2f}", style="bold cyan")
    elif intent["action"] == "trick":
        t.append("   → ", style="dim")
        t.append(f"trick: {intent['trick']}", style="bold magenta")
    else:
        t.append("   → ", style="dim")
        t.append("chat", style="bold green")
    t.append(f"   💬 {intent['reply']}", style="green")
    console.print(t)


def _wants_quit(text):
    """Quit only on a short, deliberate goodbye — not mid-sentence mentions."""
    t = text.lower()
    return any(w in t for w in QUIT_WORDS) and len(t.split()) <= 4


def voice_loop(ctrl):
    voice.speak("Robo pup two point oh online! I can walk, turn, sit, wave, dance, and fetch.")
    while True:
        console.print("[dim]listening…[/dim]", end="\r")
        text = voice.listen_until_silence(verbose=False)
        if not text:
            continue
        if _wants_quit(text):
            console.print("[bold red]powering down 🔌[/bold red]")
            voice.speak("Powering down. Goodbye, friend!")
            os._exit(0)

        intent = nlu.parse_command(text, current_vx=ctrl.speed, current_yaw=ctrl.turn)
        status(text, intent)

        if intent["action"] == "move":
            ctrl.set_command(intent["vx"], intent["yaw"])
        elif intent["action"] == "trick":
            if intent["trick"] == "fetch":
                ctrl.fetch()
            elif intent["trick"] == "spin":
                ctrl.spin()
            else:
                ctrl.do_trick(intent["trick"])
        # chat → no movement, just the reply

        voice.speak(intent["reply"])


if __name__ == "__main__":
    print("loading policy + speech model…")
    ctrl = RobotController()
    voice._get_model()

    def on_event(event):
        if event == "fetched":
            console.print("[bold green]🎾 got the ball![/bold green]")
            threading.Thread(target=lambda: voice.speak("Got it! Woof woof!"),
                             daemon=True).start()
    ctrl.on_event = on_event

    banner()
    threading.Thread(target=lambda: voice_loop(ctrl), daemon=True).start()
    ctrl.run_viewer()
