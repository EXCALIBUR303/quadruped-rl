"""Shared fixtures. Keeps MuJoCo model construction out of every test."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brain"))

_ROOT = os.path.join(os.path.dirname(__file__), "..")
# Integration tests use the pretrained policy shipped in models/final rather
# than checkpoints_v2/, which may be mid-write while a training run is active.
POLICY_DIRS = [os.path.join(_ROOT, "models", "final"),
               os.path.join(_ROOT, "checkpoints_v2")]


def find_policy():
    for d in POLICY_DIRS:
        model = os.path.join(d, "best_model.zip")
        norm  = os.path.join(d, "vec_normalize.pkl")
        if os.path.exists(model) and os.path.exists(norm):
            return model[: -len(".zip")], norm
    return None


requires_checkpoint = pytest.mark.skipif(
    find_policy() is None,
    reason="needs a trained policy (models/final or checkpoints_v2)",
)


@pytest.fixture(scope="module")
def controller():
    """A RobotController on a stable policy, with the viewer never opened."""
    from controller import RobotController
    model, norm = find_policy()
    return RobotController(model_path=model, norm_path=norm)


@pytest.fixture(scope="module")
def env():
    """A clean env with randomization OFF — deterministic for assertions."""
    from sim.envs import RobotV2Env
    from sim.config import EnvCfg
    cfg = EnvCfg()
    cfg.domain_rand.enabled = False
    cfg.obs_noise.enabled = False
    e = RobotV2Env(cfg=cfg)
    yield e
    e.close()


@pytest.fixture(scope="module")
def random_env():
    """Env with domain randomization and sensor noise ON (training config)."""
    from sim.envs import RobotV2Env
    e = RobotV2Env()
    yield e
    e.close()
