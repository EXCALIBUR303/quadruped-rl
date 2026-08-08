"""
Startup checks with actionable messages.

Before this, the failure modes were all raw tracebacks: a missing checkpoint
gave `FileNotFoundError: best_model.zip`, a stopped Ollama server surfaced as a
`ConnectionError` mid-conversation, and a blocked microphone simply produced
silence forever. Each check below tells the user exactly which command fixes it.
"""

import os
import shutil
import subprocess
import sys

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints_v2")
PRETRAINED = os.path.join(os.path.dirname(__file__), "..", "models", "final")


class PreflightError(RuntimeError):
    """Raised with a message the user can act on directly."""


# ------------------------------------------------------------------ checkpoint
def check_policy(ckpt_dir=CKPT_DIR):
    model = os.path.join(ckpt_dir, "best_model.zip")
    norm  = os.path.join(ckpt_dir, "vec_normalize.pkl")
    if os.path.exists(model) and os.path.exists(norm):
        return model[: -len(".zip")], norm

    have_pretrained = os.path.exists(os.path.join(PRETRAINED, "best_model.zip"))
    hint = (
        "\n  Use the pretrained policy shipped with this repo:\n"
        "      mkdir -p checkpoints_v2 && cp models/final/* checkpoints_v2/\n"
        if have_pretrained else
        "\n  Train one (~50 min):\n"
        "      caffeinate -i python training/train_v2.py\n"
    )
    raise PreflightError(
        f"No trained policy found in {os.path.relpath(ckpt_dir)}/." + hint +
        "\n  Check the pipeline first (~5 s):\n"
        "      python training/train_v2.py --smoke\n"
    )


# ---------------------------------------------------------------------- ollama
def check_ollama(model="llama3.2:3b", required=False):
    """Return (ok, message). The voice layer degrades to keyword parsing when
    Ollama is unavailable, so this is a warning unless `required`."""
    if shutil.which("ollama") is None:
        msg = ("Ollama is not installed — falling back to keyword commands.\n"
               "  Install:  brew install ollama && brew services start ollama\n"
               f"  Then:     ollama pull {model}")
        if required:
            raise PreflightError(msg)
        return False, msg
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                             timeout=10)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or "ollama list failed")
        if model.split(":")[0] not in out.stdout:
            msg = (f"Ollama is running but {model} is missing — using keyword "
                   f"commands.\n  Fix:  ollama pull {model}")
            if required:
                raise PreflightError(msg)
            return False, msg
        return True, f"Ollama ready ({model})"
    except Exception as exc:
        msg = (f"Ollama is installed but not responding ({exc}) — falling back "
               "to keyword commands.\n  Fix:  brew services start ollama")
        if required:
            raise PreflightError(msg)
        return False, msg


# ----------------------------------------------------------------- microphone
def check_microphone():
    try:
        import sounddevice as sd
        if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
            return False, ("No microphone detected. Connect one, or drive the "
                           "robot with the keyboard:\n"
                           "      mjpython brain/controller.py")
        return True, "Microphone ready"
    except Exception as exc:
        return False, (f"Audio system unavailable ({exc}).\n"
                       "  macOS may need mic permission: System Settings → "
                       "Privacy & Security → Microphone → enable Terminal.")


# --------------------------------------------------------------------- camera
def check_camera(index=0):
    try:
        import cv2
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if not ok:
            return False, ("Could not read from the webcam.\n"
                           "  Grant access: System Settings → Privacy & Security "
                           "→ Camera → enable Terminal, then re-run.")
        return True, "Camera ready"
    except ImportError:
        return False, "opencv-python is not installed.\n  Fix:  pip install opencv-python"
    except Exception as exc:
        return False, f"Camera unavailable ({exc})."


# ------------------------------------------------------------------ mjpython
def check_viewer_runtime():
    """The MuJoCo interactive viewer requires `mjpython` on macOS.

    Detected the way MuJoCo itself does it — via the `_MJPYTHON` object the
    launcher installs in `mujoco.viewer`. Checking `sys.executable` does NOT
    work: mjpython is a launcher that runs the normal interpreter, so
    sys.executable is still .../python3.11 and the check rejects a perfectly
    correct `mjpython script.py` invocation.
    """
    if sys.platform != "darwin":
        return True, "Viewer runtime OK"
    try:
        import mujoco.viewer
    except Exception as exc:
        return False, f"MuJoCo viewer unavailable ({exc})."

    missing = object()
    mjpy = getattr(mujoco.viewer, "_MJPYTHON", missing)
    if mjpy is missing:
        # MuJoCo internals changed; don't block — it raises its own clear error.
        return True, "Viewer runtime: could not verify, continuing"
    if mjpy is None:
        script = os.path.relpath(sys.argv[0]) if sys.argv and sys.argv[0] else "<script>"
        return False, (f"On macOS the viewer needs mjpython:\n"
                       f"      mjpython {script}")
    return True, "Viewer runtime OK (mjpython)"


def report(checks, console=None):
    """Print check results; returns True if all passed."""
    ok_all = True
    for ok, msg in checks:
        ok_all &= ok
        prefix = "  ✓ " if ok else "  ! "
        line = prefix + msg.replace("\n", "\n    ")
        if console:
            console.print(line, style="green" if ok else "yellow")
        else:
            print(line)
    return ok_all
