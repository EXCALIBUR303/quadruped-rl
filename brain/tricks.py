"""
Trick engine — choreographed motion primitives for the 12-joint robot.

Tricks are keyframe pose sequences executed through the same PD controller the
RL policy uses (this is how real robot demos choreograph moves — no RL needed).
Each trick compiles to an array of PD target poses at 50 Hz with smooth
cosine interpolation between keyframes, always ending back at the standing
pose so control can hand back to the walking policy.

Joint order (12): [hip, shoulder, knee] x [FL, FR, RL, RR]
Standing pose:    hip=0, shoulder=+0.63, knee=-1.25
Joint limits:     hip ±0.6, shoulder ±1.3, knee [-2.0, +0.3]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.config import DEFAULT_POSE, CTRL_HZ

# The standing pose is the same one the RL policy's PD controller targets at
# action == 0, so tricks blend seamlessly back into walking.
STAND = DEFAULT_POSE.copy()


def _pose(fl=None, fr=None, rl=None, rr=None):
    """Build a 12-d pose from per-leg [hip, shoulder, knee] (default: stand)."""
    default = list(STAND[:3])
    legs = [list(fl or default), list(fr or default),
            list(rl or default), list(rr or default)]
    return np.array([j for leg in legs for j in leg])


# ---------------------------------------------------------------- keyframes
SIT = _pose(
    fl=[0.0, 0.15, -0.75],       # front legs upright, chest up
    fr=[0.0, 0.15, -0.75],
    rl=[0.0, 1.30, -2.00],       # rear legs folded under the body
    rr=[0.0, 1.30, -2.00],
)

LIE = _pose(
    fl=[0.0, 1.30, -2.00],
    fr=[0.0, 1.30, -2.00],
    rl=[0.0, 1.30, -2.00],
    rr=[0.0, 1.30, -2.00],
)

BOW = _pose(                      # play bow: chest down, butt up
    fl=[0.0, 1.05, -1.80],        # front folded but not past the tipping point
    fr=[0.0, 1.05, -1.80],
    rl=[0.0, 0.50, -1.00],        # rear slightly crouched for stability
    rr=[0.0, 0.50, -1.00],
)

# wave: sitting, right-front paw raised; hip swings it side to side
_WAVE_UP = _pose(
    fl=[0.0, 0.10, -0.60],        # support leg fairly straight
    fr=[0.0, 1.30, -0.90],        # paw raised forward
    rl=[0.0, 1.30, -2.00],
    rr=[0.0, 1.30, -2.00],
)


def _wave_frames():
    """Sit → raise paw → wag it via the hip joint → sit → stand."""
    segs = [(STAND, 0.3), (SIT, 0.8), (_WAVE_UP, 0.6)]
    for k in range(4):                      # 4 side-to-side wags
        p = _WAVE_UP.copy()
        p[3] = 0.45 if k % 2 == 0 else -0.45   # FR hip
        segs.append((p, 0.30))
    segs += [(_WAVE_UP, 0.3), (SIT, 0.6), (STAND, 0.8)]
    return segs


def _dance_frames():
    """Rhythmic body groove: gentle hip sways + shoulder bobs + knee pulses."""
    segs = [(STAND, 0.3)]
    for k in range(6):
        sway = 0.18 if k % 2 == 0 else -0.18   # gentle — big sways topple it
        bob  = 0.55 if k % 2 == 0 else 0.75
        kn   = -1.15 if k % 2 == 0 else -1.35
        p = _pose(fl=[sway, bob, kn], fr=[sway, bob, kn],
                  rl=[sway, bob, kn], rr=[sway, bob, kn])
        segs.append((p, 0.40))
    # finish with a little bounce
    crouch = _pose(fl=[0, 0.85, -1.60], fr=[0, 0.85, -1.60],
                   rl=[0, 0.85, -1.60], rr=[0, 0.85, -1.60])
    segs += [(crouch, 0.3), (STAND, 0.4), (crouch, 0.3), (STAND, 0.6)]
    return segs


def _wiggle_frames():
    """Short happy butt-wiggle — celebration after fetching."""
    segs = [(STAND, 0.2)]
    for k in range(4):
        sway = 0.16 if k % 2 == 0 else -0.16
        p = _pose(fl=[sway, 0.63, -1.25], fr=[sway, 0.63, -1.25],
                  rl=[sway, 0.63, -1.25], rr=[sway, 0.63, -1.25])
        segs.append((p, 0.22))
    segs.append((STAND, 0.4))
    return segs


TRICKS = {
    "sit":    [(STAND, 0.3), (SIT, 0.9), (SIT, 1.6), (STAND, 0.9)],
    "lie":    [(STAND, 0.3), (SIT, 0.7), (LIE, 0.9), (LIE, 1.5), (SIT, 0.8), (STAND, 0.9)],
    "bow":    [(STAND, 0.3), (BOW, 0.8), (BOW, 1.4), (STAND, 0.8)],
    "wave":   _wave_frames(),
    "dance":  _dance_frames(),
    "wiggle": _wiggle_frames(),
}


def compile_trick(name, start_pose=None):
    """Compile a trick to a (T, 12) array of PD targets at CTRL_HZ.

    Cosine-interpolates from each pose to the next (starting from
    `start_pose`, default standing) for smooth, natural motion.
    """
    segs = TRICKS[name]
    targets = []
    prev = np.asarray(start_pose, dtype=float) if start_pose is not None else STAND.copy()
    for pose, dur in segs:
        n = max(2, int(dur * CTRL_HZ))
        for i in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / n)   # cosine ease
            targets.append((1 - a) * prev + a * pose)
        prev = pose.copy()
    return np.array(targets)


def available():
    return sorted(TRICKS.keys())
