# 🐕 Voice-Controlled Quadruped Robot Dog (RL in Simulation)

Train a four-legged robot to walk and turn in physics simulation with deep
reinforcement learning, then **talk to it** — speak a command, it obeys in the
viewer and replies out loud.

Everything runs locally on a Mac. No robot hardware required.

```
🎤 "walk forward and turn left"
      │
      ▼  Whisper (speech-to-text)
   "walk forward and turn left"
      │
      ▼  Ollama LLM (intent → command)
   { vx: 0.6, yaw: 0.6 }
      │
      ▼  PPO policy (trained in MuJoCo)
   12 joint targets → robot trots and curves left
      │
      ▼  macOS `say` (text-to-speech)
🔊 "Curving off to the left!"
```

## What it does

- A **Unitree Go2** quadruped (12 joints) trained from scratch with PPO to
  follow velocity + turn commands: stand, walk at any speed, reverse, turn.
- A **voice layer** that turns natural speech into robot commands and talks
  back — all with local, offline models.

## Tech stack

| Part | Tool |
|------|------|
| Physics sim | [MuJoCo](https://mujoco.org) + [Unitree Go2 model](https://github.com/google-deepmind/mujoco_menagerie) |
| RL | [Gymnasium](https://gymnasium.farama.org) + [Stable-Baselines3](https://stable-baselines3.readthedocs.io) (PPO) |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Intent / chat | [Ollama](https://ollama.com) (llama3.2:3b), local & offline |
| Text-to-speech | macOS built-in `say` |

## Project layout

```
sim/
  envs/go2_env.py        # legged_gym-style Go2 environment (PD control, 48-d obs)
  envs/quadruped_env.py  # earlier hand-built 8-DOF robot (see "The journey")
  assets/go2/            # Unitree Go2 MuJoCo model
training/
  train_go2.py           # PPO training (8 parallel envs)
  evaluate_go2.py        # command-tracking metrics (speed + turn)
brain/
  controller.py          # loads policy, runs viewer, live set_command(vx, yaw)
  nlu.py                 # speech text → {vx, yaw, reply} via Ollama
  voice.py               # mic → text (Whisper) + spoken replies (say)
  talk_to_robot.py       # full pipeline: speak → understand → obey → reply
```

## Quick start

```bash
# 1. setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. train the robot to walk + turn (background, ~1.5h on an M-series Mac)
python training/train_go2.py

# 3. check how well it follows commands
python training/evaluate_go2.py

# 4. talk to it!  (needs Ollama running: `ollama serve`, and `ollama pull llama3.2:3b`)
mjpython brain/talk_to_robot.py        # macOS: mjpython, not python
```

Try saying: *"walk forward"*, *"go faster"*, *"turn left"*, *"back up"*, *"stop"*.

Prefer keyboard? `mjpython brain/controller.py` (type a speed, `l`/`r` to turn, `s` stop).

## The journey

This project was built iteratively, and the dead-ends are part of the story:

1. **Hand-built 8-DOF robot** (2 joints/leg). Learned a clean forward trot and
   variable-speed control — but **could not turn**. Root cause: with only
   fore/aft leg joints and no hip-abduction joint, the body physically cannot
   yaw without destabilising. This is a *morphology* limit, not a tuning one.
2. **Upgraded to the Unitree Go2** (12 joints, 3/leg) — the real robot-dog
   design. The extra hip-abduction joint per leg is exactly what makes turning
   and a stable trot possible, which the research literature confirms.
3. **Reward design** followed the `legged_gym` recipe (exponential velocity +
   yaw tracking, foot-air-time for clean stepping, orientation/torque/smoothness
   penalties) plus a linear directional-progress term to avoid the
   "stand still and do nothing" local optimum that pure tracking rewards fall
   into with limited parallelism.

## Known limitations

- The 8-DOF robot (`quadruped_env.py`) is kept for reference; it cannot turn.
- Trained on flat ground only (no rough terrain / disturbances yet).
- Sim-only — sim-to-real transfer to physical hardware is future work.

## License

Code: MIT. The Unitree Go2 model under `sim/assets/go2/` is Apache-2.0
(© Unitree Robotics / Google DeepMind MuJoCo Menagerie).
