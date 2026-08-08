"""Run manifests — record exactly what produced a checkpoint.

Without this, a checkpoint is an anonymous binary: you cannot tell which reward
weights, which randomization ranges, or which code version created it, so a
result can't be reproduced or even explained a week later. Every training run
writes `run_manifest.json` next to its checkpoints.
"""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def collect(seed, total_steps, n_envs, env_cfg, ppo_kwargs, extra=None):
    """Build a JSON-serializable description of a training run."""
    import numpy as np
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "seed": seed,
        "total_steps": total_steps,
        "n_envs": n_envs,
        "env_cfg": env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else dict(env_cfg),
        "ppo": {k: (list(v) if isinstance(v, (tuple, list)) else v)
                for k, v in ppo_kwargs.items()},
        "platform": {
            "python": platform.python_version(),
            "system": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
    }
    try:
        import stable_baselines3, mujoco, gymnasium
        manifest["platform"].update({
            "stable_baselines3": stable_baselines3.__version__,
            "mujoco": mujoco.__version__,
            "gymnasium": gymnasium.__version__,
        })
    except Exception:
        pass
    if extra:
        manifest.update(extra)
    return manifest


def write(path, manifest):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def load(path):
    with open(path) as f:
        return json.load(f)
