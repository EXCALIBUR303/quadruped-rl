"""
Natural-language understanding for the 12-joint robot: speech → command via the
local Ollama LLM (with a keyword fallback). This robot has hip servos, so it can
walk forward/backward AND turn left/right.

Command = {"vx": forward m/s, "yaw": turn rad/s, "reply": spoken confirmation}.
  vx :  -0.3 (back) .. 0.0 (stop) .. 0.8 (fast forward)
  yaw:  -0.8 (turn right) .. 0.0 (straight) .. 0.8 (turn left)
"""

import json
import ollama

MODEL = "llama3.2:3b"
VX_LO, VX_HI   = -0.3, 0.8
YAW_LO, YAW_HI = -0.8, 0.8

SYSTEM_PROMPT = """You are the friendly brain of a four-legged robot dog. Convert the user's instruction into a JSON movement command AND a short, warm spoken reply with a bit of personality (like an eager robot pup).

Output ONLY compact JSON: {"vx": <number>, "yaw": <number>, "reply": "<short friendly confirmation>"}
- vx = forward speed m/s: stop=0.0, slow=0.4, normal=0.6, fast=0.8, backward=-0.3
- yaw = turn rate rad/s: straight=0.0, turn left=+0.6, turn right=-0.6 (sharper=bigger)
- You CAN combine moving and turning.

Examples:
"walk forward"        -> {"vx":0.6,"yaw":0.0,"reply":"On my way, trotting forward!"}
"go fast"             -> {"vx":0.8,"yaw":0.0,"reply":"Full speed, here I go!"}
"take it slow"        -> {"vx":0.4,"yaw":0.0,"reply":"Easy does it."}
"turn left"           -> {"vx":0.0,"yaw":0.6,"reply":"Turning left for you!"}
"turn right"          -> {"vx":0.0,"yaw":-0.6,"reply":"Spinning right, watch me!"}
"walk and curve left" -> {"vx":0.5,"yaw":0.4,"reply":"Curving off to the left!"}
"back up"             -> {"vx":-0.3,"yaw":0.0,"reply":"Backing up, beep beep!"}
"stop"                -> {"vx":0.0,"yaw":0.0,"reply":"Stopping right here!"}
"""


def parse_command(text, current_vx=0.0, current_yaw=0.0):
    text = (text or "").strip()
    if not text:
        return {"vx": current_vx, "yaw": current_yaw, "reply": "I didn't catch that."}
    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": text}],
            format="json", options={"temperature": 0.4},
        )
        data  = json.loads(resp["message"]["content"])
        vx    = max(VX_LO,  min(VX_HI,  float(data.get("vx", 0.0))))
        yaw   = max(YAW_LO, min(YAW_HI, float(data.get("yaw", 0.0))))
        reply = str(data.get("reply", "Okay!")) or "Okay!"
        return {"vx": vx, "yaw": yaw, "reply": reply}
    except Exception:
        return _keyword_fallback(text, current_vx, current_yaw)


def _keyword_fallback(text, current_vx, current_yaw):
    t = text.lower()
    if any(w in t for w in ["stop", "halt", "stay", "freeze", "wait"]):
        return {"vx": 0.0, "yaw": 0.0, "reply": "Stopping!"}
    if "left" in t:
        vx = 0.4 if any(w in t for w in ["walk", "forward", "go", "move"]) else 0.0
        return {"vx": vx, "yaw": 0.6, "reply": "Turning left!"}
    if "right" in t:
        vx = 0.4 if any(w in t for w in ["walk", "forward", "go", "move"]) else 0.0
        return {"vx": vx, "yaw": -0.6, "reply": "Turning right!"}
    if "back" in t or "reverse" in t:
        return {"vx": -0.3, "yaw": 0.0, "reply": "Backing up!"}
    if "faster" in t or "speed up" in t:
        return {"vx": min(VX_HI, current_vx + 0.2), "yaw": current_yaw, "reply": "Faster!"}
    if "slower" in t or "slow down" in t:
        return {"vx": max(0.0, current_vx - 0.2), "yaw": current_yaw, "reply": "Slowing down."}
    if any(w in t for w in ["fast", "quick", "run", "hurry"]):
        return {"vx": 0.8, "yaw": 0.0, "reply": "Zooming ahead!"}
    if any(w in t for w in ["slow", "easy", "gentle"]):
        return {"vx": 0.4, "yaw": 0.0, "reply": "Taking it easy."}
    if any(w in t for w in ["walk", "forward", "go", "move", "ahead", "come"]):
        return {"vx": 0.6, "yaw": 0.0, "reply": "Trotting forward!"}
    return {"vx": current_vx, "yaw": current_yaw, "reply": "Hmm, I didn't understand that."}


if __name__ == "__main__":
    for p in ["walk forward", "go really fast", "turn left", "turn right",
              "back up a bit", "walk and curve right", "stop now"]:
        c = parse_command(p)
        print(f"{p!r:34} -> vx={c['vx']:+.2f} yaw={c['yaw']:+.2f} | {c['reply']}")
