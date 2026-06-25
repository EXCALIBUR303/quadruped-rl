# 🐕 Voice-Controlled Quadruped Robot Dog (RL in Simulation)

Train a four-legged robot to walk in physics simulation with deep reinforcement
learning, then **talk to it** — speak a command, it obeys in the viewer and
replies out loud. Everything runs locally on a Mac. No robot hardware required.

```
🎤 "walk forward, but take it easy"
      │
      ▼  Whisper (speech-to-text)
   "walk forward but take it easy"
      │
      ▼  Ollama LLM (intent → command)
   { vx: 0.3 }
      │
      ▼  PPO policy (trained in MuJoCo)
   8 joint commands → robot trots forward slowly
      │
      ▼  macOS `say` (text-to-speech)
🔊 "Easy does it, nice and gentle."
```

## What it does

- A custom **8-joint quadruped** (2 motors per leg) trained from scratch with
  PPO to do **velocity-controlled walking**: stand still, and trot forward at a
  commanded speed (slow → fast), with a clean diagonal-trot gait.
- A **voice layer** that turns natural speech into robot commands and talks
  back with personality — all with local, offline models.

## Tech stack

| Part | Tool |
|------|------|
| Physics sim | [MuJoCo](https://mujoco.org) |
| RL | [Gymnasium](https://gymnasium.farama.org) + [Stable-Baselines3](https://stable-baselines3.readthedocs.io) (PPO) |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Intent / chat | [Ollama](https://ollama.com) (llama3.2:3b), local & offline |
| Text-to-speech | macOS built-in `say` |

## Project layout

```
sim/
  envs/quadruped_env.py  # the 8-joint robot — gait clock, speed-tracking reward
  envs/go2_env.py        # 12-joint Unitree Go2 (see "Exploring turning")
  assets/quadruped.xml   # hand-built 8-joint robot model
  assets/go2/            # Unitree Go2 model (MuJoCo Menagerie)
training/
  train.py               # PPO training for the 8-joint robot
  evaluate.py            # command-tracking metrics (stand / slow / fast)
brain/
  controller.py          # loads policy, runs viewer, live set_command(speed)
  nlu.py                 # speech text → {speed, reply} via Ollama
  voice.py               # mic → text (Whisper) + spoken replies (say)
  talk_to_robot.py       # full pipeline: speak → understand → obey → reply
```

## Quick start

```bash
# 1. setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. train the robot to walk (background, ~13 min on an M-series Mac)
python training/train.py

# 3. check how well it follows speed commands
python training/evaluate.py

# 4. talk to it!  (needs Ollama: `ollama serve` + `ollama pull llama3.2:3b`)
mjpython brain/talk_to_robot.py        # macOS: mjpython, not python
```

Try saying: *"walk forward"*, *"go faster"*, *"take it slow"*, *"stop"*.

Prefer keyboard? `mjpython brain/controller.py` (type a speed, `s` to stop).

## Results

- **Speed tracking 4/4**: stands still, and trots at slow / medium / fast on
  command.
- **Clean diagonal trot** — diagonal foot pairs alternate (FL+RR / FR+RL),
  ~2 feet on the ground at a time, body held level.
- **Crisp stop** — halts in ~0.6 s with negligible drift when standing.
- **Voice loop** runs fully offline: speak → it obeys → it replies with
  personality.

## The journey (and an honest engineering trade-off)

This was built iteratively, and the dead-ends are part of the story:

1. **8-joint robot, forward trot.** Fixed the leg geometry/joint axes, then
   shaped the reward through several stages: clean trot → stability → a periodic
   gait clock for a proper diagonal trot → velocity-command control.
2. **Turning?** A quadruped with only fore/aft leg joints (no hip-abduction
   joint) physically struggles to turn without destabilising — confirmed over
   many reward iterations. This is a *morphology* limit, not a tuning one.
3. **Exploring turning with the Unitree Go2** (`go2_env.py`, 12 joints). The
   extra hip-abduction joint per leg is exactly what enables turning. But
   training a 12-joint robot to walk well needs **GPU-scale simulation**
   (~thousands of robots in parallel); on a laptop CPU (8 parallel envs) it only
   reached a slow creep after 40M+ steps. The Go2 environment is included as a
   documented foundation for anyone with a GPU.

**Decision:** ship the 8-joint robot, which walks cleanly and runs the full
voice demo reliably. Turning is documented as requiring either GPU-scale
training (Go2) or a physical hip-abduction servo.

## Known limitations

- The 8-joint robot controls forward speed only — no turning (see above).
- Trained on flat ground; no rough terrain / disturbances yet.
- Sim-only — sim-to-real transfer to physical hardware is future work.

## License

Code: MIT. The Unitree Go2 model under `sim/assets/go2/` is Apache-2.0
(© Unitree Robotics / Google DeepMind MuJoCo Menagerie).
