# 🐕 Voice-Controlled Robot Dog (RL in Simulation)

Train a four-legged robot to **walk and turn** in physics simulation with deep
reinforcement learning, then **talk to it** — speak a command, it obeys in the
viewer and replies out loud. Everything runs locally on a Mac. No hardware.

![robot](docs/robot.png)

```
🎤 "walk forward, then turn left"
      │  Whisper (speech-to-text)
      ▼  Ollama LLM (intent → command)
   { vx: 0.6, yaw: 0.6 }
      │  PPO policy (trained in MuJoCo)
      ▼  12 joint commands → robot trots and curves left
🔊 "Curving off to the left!"   (macOS `say`)
```

## What it does

- A custom **12-joint robot dog** (3 motors per leg, incl. a hip-abduction
  joint) trained from scratch with PPO to follow velocity + turn commands:
  stand, walk at any speed, reverse, and **turn left/right**.
- **Tricks**: sit, lie down, wave a paw, dance, play-bow, spin — choreographed
  keyframe motions played through the same PD controller as the policy
  (the same way real robot-dog demos are choreographed), each one
  physics-verified so it never falls.
- **Fetch**: say *"go get the ball"* — a glowing ball drops into the world and
  the robot **autonomously navigates to it** (a heading P-controller steering
  the RL walking policy), then celebrates with a butt-wiggle.
- **Mood eyes**: the visor glows cyan when idle, orange during tricks, green
  while fetching.
- A **voice layer** that turns natural speech into commands and talks back with
  personality — movement, tricks, *and* free chat ("who are you?" gets an
  in-character answer). All local & offline.
- **Robustness**: Whisper silence-hallucination filtering (no more phantom
  "The The The" commands) and unbounded episodes (the robot no longer
  teleports back to origin after 20 s of demo).

## Control: PD position control

The policy does **not** output torques. It outputs **target joint angles**
(offsets from a standing pose); a PD loop (`τ = Kp·(target−q) − Kd·q̇`,
Kp=25, Kd=0.7) turns them into smooth motor torques each sim step. This is the
standard, stable way to train legged robots — raw torque control led to a
collapsed, foot-sliding gait that still tracked velocity (a trap the old
velocity-only eval failed to catch).

## Results (`evaluate_v2.py` — checks velocity **and** posture **and** foot-slip)

| Command | Asked | Measured | Height | Slip |
|---|---|---|---|---|
| stand still | 0 | 0.00 | 0.27 | 0.02 ✅ |
| walk slow | 0.3 m/s | 0.31 | 0.28 | 0.20 ✅ |
| walk fast | 0.6 m/s | 0.62 | 0.28 | 0.29 ◑ |
| turn left | +0.6 rad/s | +0.64 | 0.27 | 0.16 ✅ |
| turn right | −0.6 rad/s | −0.67 | 0.27 | 0.12 ✅ |
| walk + turn | 0.4 / 0.4 | 0.40 / 0.33 | 0.28 | 0.27 ◑ |

Stands tall (0.27 m), turns cleanly both ways, walks at the commanded speed.
Forward foot-slip is low (clean step at slow speed; a slight drag at top
speed, ◑). The eval deliberately scores foot-slip and height so a collapsed
or sliding policy cannot pass on velocity alone. See `robot_demo.mp4`.

## Tech stack

| Part | Tool |
|------|------|
| Physics sim | [MuJoCo](https://mujoco.org) |
| RL | [Gymnasium](https://gymnasium.farama.org) + [Stable-Baselines3](https://stable-baselines3.readthedocs.io) (PPO) |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Intent / chat | [Ollama](https://ollama.com) (llama3.2:3b), local & offline |
| Text-to-speech | macOS built-in `say` |

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train the 12-joint robot to walk + turn (resume-capable, ~40 min)
caffeinate -i python training/train_v2.py

python training/evaluate_v2.py     # check walk + turn tracking
python scripts/render_walk.py      # render robot_demo.mp4

# talk to it!  (needs Ollama: `brew services start ollama` + `ollama pull llama3.2:3b`)
mjpython brain/talk_to_robot.py    # macOS: mjpython, not python
```

Try: *"walk forward"*, *"turn left"*, *"sit"*, *"wave hello"*, *"dance"*,
*"take a bow"*, *"spin around"*, *"go fetch the ball"*, *"stop"*, *"goodbye"* —
or just chat with it.

Keyboard alternative: `mjpython brain/controller.py`
(number=speed · l/r=turn · s=stop · `t sit|wave|dance|bow|lie` · f=fetch · p=spin).

## Project layout

```
sim/
  assets/robot_v2.xml    # the 12-joint robot dog (sleek robot styling)
  envs/robot_v2_env.py   # 39-d obs, 12 actions, gait clock + velocity/turn reward
training/
  train_v2.py            # PPO training (resume-capable)
  evaluate_v2.py         # walk + turn command tracking
scripts/render_walk.py   # renders the showcase video (walk, tricks, fetch)
brain/
  controller.py          # mode machine: POLICY walk/turn · TRICK · NAVIGATE (fetch) · SPIN
  tricks.py              # keyframe choreography engine (sit/lie/wave/dance/bow/wiggle)
  nlu.py                 # speech → {action, vx, yaw, trick, reply} via Ollama
  voice.py               # Whisper STT (w/ hallucination filter) + `say` TTS
  talk_to_robot.py       # full pipeline + console UI
```

## The journey

Built iteratively — the dead-ends are part of the story:

1. **8-joint robot** (`quadruped_env.py`) — learned a clean forward trot and
   speed control, but **physically can't turn** (no sideways hip joint).
2. **Unitree Go2** (`go2_env.py`, 12 joints) — can turn, but training a heavy,
   realistic robot needs **GPU-scale simulation**; on a laptop CPU it only
   crept. Kept as a GPU-ready foundation.
3. **Custom 12-joint robot** (`robot_v2.xml`) — added hip-abduction servos to
   the *lightweight* robot. Same fast-training physics that walked in minutes,
   now with the joints needed to turn. **Walks and turns on a laptop CPU.** ✅

## License

Code: MIT. Unitree Go2 model under `sim/assets/go2/`: Apache-2.0
(© Unitree Robotics / Google DeepMind MuJoCo Menagerie).
