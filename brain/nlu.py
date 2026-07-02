"""
Natural-language understanding for the robot pup: speech → structured intent
via the local Ollama LLM, with a keyword fallback.

Intent = {"action": "move" | "trick" | "chat",
          "vx": float, "yaw": float,        # for move
          "trick": str,                     # for trick: sit lie wave dance bow spin fetch
          "reply": str}                     # always: what the pup says out loud

The robot can walk forward/backward, turn left/right, and perform tricks.
Anything that isn't a movement request is a chat — answer in character.
"""

import json
import ollama

MODEL = "llama3.2:3b"
VX_LO, VX_HI   = -0.3, 0.6
YAW_LO, YAW_HI = -0.7, 0.7
TRICK_NAMES = {"sit", "lie", "wave", "dance", "bow", "spin", "fetch", "patrol", "follow"}

SYSTEM_PROMPT = """You are ROBO-PUP, a cheerful four-legged robot dog. Convert the user's words into a JSON intent AND a short warm spoken reply (eager-puppy personality, max ~12 words).

Output ONLY compact JSON: {"action":"move"|"trick"|"chat", "vx":<num>, "yaw":<num>, "trick":"<name>", "reply":"<speech>"}

MOVE — walking/turning. vx m/s: stop=0, slow=0.3, normal=0.45, fast=0.6, backward=-0.3.
       yaw rad/s: straight=0, left=+0.6, right=-0.6. Can combine.
TRICK — one of: sit, lie, wave, dance, bow, spin, fetch, patrol, follow.
        patrol = walk a loop around the arena. follow = follow the ball around.
CHAT — questions/small talk: no movement, just answer in character (vx=0,yaw=0).

Examples:
"walk forward"        -> {"action":"move","vx":0.45,"yaw":0.0,"trick":"","reply":"On my way, trotting forward!"}
"turn left"           -> {"action":"move","vx":0.0,"yaw":0.6,"trick":"","reply":"Turning left!"}
"stop"                -> {"action":"move","vx":0.0,"yaw":0.0,"trick":"","reply":"Stopping right here!"}
"sit down"            -> {"action":"trick","vx":0,"yaw":0,"trick":"sit","reply":"Sitting like a good pup!"}
"can you dance"       -> {"action":"trick","vx":0,"yaw":0,"trick":"dance","reply":"Watch my moves!"}
"wave hello"          -> {"action":"trick","vx":0,"yaw":0,"trick":"wave","reply":"Hi hi hi! Waving!"}
"go get the ball"     -> {"action":"trick","vx":0,"yaw":0,"trick":"fetch","reply":"Ball! Ball! On it!"}
"lie down"            -> {"action":"trick","vx":0,"yaw":0,"trick":"lie","reply":"Flopping down for a rest."}
"spin around"         -> {"action":"trick","vx":0,"yaw":0,"trick":"spin","reply":"Wheee, spinning!"}
"patrol the area"     -> {"action":"trick","vx":0,"yaw":0,"trick":"patrol","reply":"On guard duty! Patrolling."}
"follow the ball"     -> {"action":"trick","vx":0,"yaw":0,"trick":"follow","reply":"I'll chase it everywhere!"}
"who are you?"        -> {"action":"chat","vx":0,"yaw":0,"trick":"","reply":"I'm Robo-Pup, your robot best friend!"}
"good boy"            -> {"action":"chat","vx":0,"yaw":0,"trick":"","reply":"Wag wag! I try my best!"}
"""


def parse_command(text, current_vx=0.0, current_yaw=0.0):
    text = (text or "").strip()
    if not text:
        return {"action": "chat", "vx": current_vx, "yaw": current_yaw,
                "trick": "", "reply": "I didn't catch that."}
    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": text}],
            format="json", options={"temperature": 0.4},
        )
        data   = json.loads(resp["message"]["content"])
        action = str(data.get("action", "chat")).lower()
        trick  = str(data.get("trick", "")).lower().strip()
        vx     = max(VX_LO,  min(VX_HI,  float(data.get("vx", 0.0) or 0.0)))
        yaw    = max(YAW_LO, min(YAW_HI, float(data.get("yaw", 0.0) or 0.0)))
        reply  = str(data.get("reply", "Okay!")) or "Okay!"
        if action == "trick" and trick not in TRICK_NAMES:
            action = "chat"
        if action not in ("move", "trick", "chat"):
            action = "chat"
        return {"action": action, "vx": vx, "yaw": yaw, "trick": trick, "reply": reply}
    except Exception:
        return _keyword_fallback(text, current_vx, current_yaw)


def _keyword_fallback(text, current_vx, current_yaw):
    t = text.lower()

    def move(vx, yaw, reply):
        return {"action": "move", "vx": vx, "yaw": yaw, "trick": "", "reply": reply}

    def trick(name, reply):
        return {"action": "trick", "vx": 0, "yaw": 0, "trick": name, "reply": reply}

    if "patrol" in t or "guard" in t:
        return trick("patrol", "Patrolling the arena!")
    if "follow" in t or "chase" in t:
        return trick("follow", "Following the ball!")
    if any(w in t for w in ["fetch", "ball", "go get"]):
        return trick("fetch", "Ball! On it!")
    if "sit" in t:
        return trick("sit", "Sitting!")
    if "lie" in t or "lay down" in t:
        return trick("lie", "Lying down.")
    if "wave" in t or "hello" in t or "hi " in t:
        return trick("wave", "Waving hi!")
    if "danc" in t:
        return trick("dance", "Dance time!")
    if "bow" in t or "stretch" in t:
        return trick("bow", "Stretching!")
    if "spin" in t or "circle" in t:
        return trick("spin", "Spinning!")
    if any(w in t for w in ["stop", "halt", "stay", "freeze", "wait"]):
        return move(0.0, 0.0, "Stopping!")
    if "left" in t:
        vx = 0.4 if any(w in t for w in ["walk", "forward", "go", "move"]) else 0.0
        return move(vx, 0.6, "Turning left!")
    if "right" in t:
        vx = 0.4 if any(w in t for w in ["walk", "forward", "go", "move"]) else 0.0
        return move(vx, -0.6, "Turning right!")
    if "back" in t or "reverse" in t:
        return move(-0.3, 0.0, "Backing up!")
    if "faster" in t or "speed up" in t:
        return move(min(VX_HI, current_vx + 0.15), current_yaw, "Faster!")
    if "slower" in t or "slow down" in t:
        return move(max(0.0, current_vx - 0.15), current_yaw, "Slowing down.")
    if any(w in t for w in ["fast", "quick", "run", "hurry"]):
        return move(0.6, 0.0, "Zooming!")
    if any(w in t for w in ["slow", "easy", "gentle"]):
        return move(0.3, 0.0, "Taking it easy.")
    if any(w in t for w in ["walk", "forward", "go", "move", "ahead", "come"]):
        return move(0.45, 0.0, "Trotting forward!")
    return {"action": "chat", "vx": current_vx, "yaw": current_yaw, "trick": "",
            "reply": "Woof! Tell me to walk, turn, sit, dance, wave, or fetch!"}


if __name__ == "__main__":
    for p in ["walk forward", "turn left", "sit down", "can you dance for me",
              "wave hello", "go fetch the ball", "spin around", "stop",
              "who are you?", "you're a good boy"]:
        c = parse_command(p)
        tag = c["trick"] if c["action"] == "trick" else f"vx={c['vx']:+.2f} yaw={c['yaw']:+.2f}"
        print(f"{p!r:28} -> {c['action']:5s} {tag:18s} | {c['reply']}")
