"""
Natural-language understanding: turn a spoken instruction into a robot
command, using the local Ollama LLM with a keyword fallback.

A command is {"vx": forward_speed_m_s, "yaw": 0.0, "reply": spoken_confirmation}.
The robot can only control forward speed (0.0-0.7 m/s) and stop — it cannot
turn or reverse (hardware limitation: no hip-abduction servo).
"""

import json
import ollama

MODEL    = "llama3.2:3b"
VX_MIN   = 0.0
VX_MAX   = 0.7

SYSTEM_PROMPT = """You control a four-legged robot. Convert the user's instruction into a JSON command.
The robot can ONLY walk forward at a chosen speed, or stop. It CANNOT turn, strafe, or go backward.
Output ONLY compact JSON: {"vx": <number 0.0-0.7>, "reply": "<very short spoken confirmation>"}
vx is forward speed in metres/second. Guide: stop=0.0, slow=0.3, normal=0.45, fast=0.7.
Examples:
"walk forward"      -> {"vx": 0.45, "reply": "Walking forward."}
"go fast"           -> {"vx": 0.7,  "reply": "Speeding up."}
"take it easy"      -> {"vx": 0.3,  "reply": "Walking slowly."}
"stop" / "halt"     -> {"vx": 0.0,  "reply": "Stopping."}
"can you turn left" -> {"vx": 0.45, "reply": "I can't turn, but I'll walk forward."}
"""


def parse_command(text, current_vx=0.0):
    """Return {'vx', 'yaw', 'reply'} for a natural-language instruction."""
    text = (text or "").strip()
    if not text:
        return {"vx": current_vx, "yaw": 0.0, "reply": "I didn't hear anything."}
    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": text}],
            format="json",
            options={"temperature": 0},
        )
        data  = json.loads(resp["message"]["content"])
        vx    = max(VX_MIN, min(VX_MAX, float(data.get("vx", 0.0))))
        reply = str(data.get("reply", "Okay.")) or "Okay."
        return {"vx": vx, "yaw": 0.0, "reply": reply}
    except Exception:
        return _keyword_fallback(text, current_vx)


def _keyword_fallback(text, current_vx):
    t = text.lower()
    if any(w in t for w in ["stop", "halt", "stay", "freeze", "wait", "stand"]):
        return {"vx": 0.0, "yaw": 0.0, "reply": "Stopping."}
    if "faster" in t or "speed up" in t:
        return {"vx": min(VX_MAX, current_vx + 0.2), "yaw": 0.0, "reply": "Faster."}
    if "slower" in t or "slow down" in t:
        return {"vx": max(VX_MIN, current_vx - 0.2), "yaw": 0.0, "reply": "Slowing down."}
    if any(w in t for w in ["fast", "quick", "run", "hurry"]):
        return {"vx": 0.7, "yaw": 0.0, "reply": "Going fast."}
    if any(w in t for w in ["slow", "easy", "gentle", "careful"]):
        return {"vx": 0.3, "yaw": 0.0, "reply": "Walking slowly."}
    if any(w in t for w in ["walk", "forward", "go", "move", "ahead", "come"]):
        return {"vx": 0.45, "yaw": 0.0, "reply": "Walking forward."}
    if any(w in t for w in ["turn", "left", "right", "back"]):
        return {"vx": 0.45, "yaw": 0.0, "reply": "I can only walk forward, but here I go."}
    return {"vx": current_vx, "yaw": 0.0, "reply": "Sorry, I didn't understand."}


if __name__ == "__main__":
    # quick manual test
    for phrase in ["walk forward", "go really fast", "take it slow please",
                   "stop now", "turn left", "can you move ahead a bit"]:
        cmd = parse_command(phrase)
        print(f"{phrase!r:40} -> vx={cmd['vx']:.2f}  reply={cmd['reply']!r}")
