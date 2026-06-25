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
- A **voice layer** that turns natural speech into commands and talks back with
  personality — all local & offline.

## Results (`evaluate_v2.py`)

| Command | Asked | Measured |
|---|---|---|
| stand still | 0 | 0.00 ✅ |
| walk slow | 0.4 m/s | 0.39 ✅ |
| walk fast | 0.8 m/s | 0.85 ✅ |
| turn left | +0.6 rad/s | +0.36 ✅ |
| turn right | −0.6 rad/s | −0.40 ✅ |
| walk + turn | 0.4 / 0.4 | 0.16 / 0.25 ✅ |

**6/6.** Clean trot (~2.5 feet on the ground, body held level within ±2°),
turns both directions. See `robot_demo.mp4`.

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

Try: *"walk forward"*, *"turn left"*, *"go faster"*, *"back up"*, *"stop"*.
Keyboard alternative: `mjpython brain/controller.py` (number=speed, l/r=turn, s=stop).

## Project layout

```
sim/
  assets/robot_v2.xml    # the 12-joint robot dog (sleek robot styling)
  envs/robot_v2_env.py   # 39-d obs, 12 actions, gait clock + velocity/turn reward
training/
  train_v2.py            # PPO training (resume-capable)
  evaluate_v2.py         # walk + turn command tracking
scripts/render_walk.py   # renders the demo video
brain/
  controller.py          # loads policy, runs viewer, live set_command(vx, yaw)
  nlu.py                 # speech → {vx, yaw, reply} via Ollama
  voice.py               # mic → text (Whisper) + spoken replies (say), robot voice
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
