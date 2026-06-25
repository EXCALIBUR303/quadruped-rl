"""
Natural-language understanding: turn a spoken instruction into a robot command,
using the local Ollama LLM with a keyword fallback.

This 8-joint robot controls FORWARD SPEED only (it has no sideways hip joint,
so it can't turn). Command = {"vx": forward m/s 0.0-0.7, "yaw": 0.0, "reply"}.
"""

import json
import ollama

MODEL  = "llama3.2:3b"
VX_LO, VX_HI = 0.0, 0.7

SYSTEM_PROMPT = """You are the friendly brain of a four-legged robot dog. Convert the user's instruction into a JSON movement command AND a short, warm spoken reply with personality (like an eager puppy).

IMPORTANT: this robot can ONLY walk forward at a chosen speed, or stop. It CANNOT turn, strafe, or go backward. If asked to turn/reverse, set a forward speed and cheerfully explain you can only go forward.

Output ONLY compact JSON: {"vx": <number 0.0-0.7>, "reply": "<short friendly confirmation>"}
- vx is forward speed in m/s: stop=0.0, slow=0.3, normal=0.45, fast=0.7

Examples:
"walk forward"   -> {"vx":0.45,"reply":"On my way! Trotting forward."}
"go fast"        -> {"vx":0.7,"reply":"Full speed, here I go!"}
"take it slow"   -> {"vx":0.3,"reply":"Easy does it, nice and gentle."}
"stop"           -> {"vx":0.0,"reply":"Stopping right here. Good boy, right?"}
"turn left"      -> {"vx":0.45,"reply":"I can't turn yet, but I'll trot straight ahead!"}
"go backward"    -> {"vx":0.0,"reply":"I can only go forward, so I'll just wait here!"}
"""


def parse_command(text, current_vx=0.0, current_yaw=0.0):
    text = (text or "").strip()
    if not text:
        return {"vx": current_vx, "yaw": 0.0, "reply": "I didn't catch that."}
    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": text}],
            format="json",
            options={"temperature": 0.4},
        )
        data  = json.loads(resp["message"]["content"])
        vx    = max(VX_LO, min(VX_HI, float(data.get("vx", 0.0))))
        reply = str(data.get("reply", "Okay!")) or "Okay!"
        return {"vx": vx, "yaw": 0.0, "reply": reply}
    except Exception:
        return _keyword_fallback(text, current_vx)


def _keyword_fallback(text, current_vx):
    t = text.lower()
    if any(w in t for w in ["stop", "halt", "stay", "freeze", "wait"]):
        return {"vx": 0.0, "yaw": 0.0, "reply": "Stopping!"}
    if any(w in t for w in ["turn", "left", "right", "back", "reverse"]):
        return {"vx": 0.45, "yaw": 0.0, "reply": "I can only go forward, so off I trot!"}
    if "faster" in t or "speed up" in t:
        return {"vx": min(VX_HI, current_vx + 0.2), "yaw": 0.0, "reply": "Faster!"}
    if "slower" in t or "slow down" in t:
        return {"vx": max(VX_LO, current_vx - 0.2), "yaw": 0.0, "reply": "Slowing down."}
    if any(w in t for w in ["fast", "quick", "run", "hurry"]):
        return {"vx": 0.7, "yaw": 0.0, "reply": "Zooming ahead!"}
    if any(w in t for w in ["slow", "easy", "gentle"]):
        return {"vx": 0.3, "yaw": 0.0, "reply": "Taking it easy."}
    if any(w in t for w in ["walk", "forward", "go", "move", "ahead", "come"]):
        return {"vx": 0.45, "yaw": 0.0, "reply": "Trotting forward!"}
    return {"vx": current_vx, "yaw": 0.0, "reply": "Hmm, I didn't understand that."}


if __name__ == "__main__":
    for phrase in ["walk forward", "go really fast", "take it slow please",
                   "stop now", "turn left", "can you move ahead a bit"]:
        c = parse_command(phrase)
        print(f"{phrase!r:34} -> vx={c['vx']:.2f} | {c['reply']}")
