"""Trick choreography: schema + physics safety.

Tricks are open-loop keyframe sequences, so a bad pose is not caught by the RL
policy — it just tips the robot over. Two tricks (bow, dance) genuinely did
this during development. These tests replay every trick through the real PD
controller and assert the robot survives, so a future pose edit can't silently
reintroduce a faceplant.
"""

import numpy as np
import pytest
import mujoco

import tricks
from sim.config import (DEFAULT_POSE, TORQUE_LIMIT, KP, KD, N_JOINTS,
                        N_FJ_Q, N_FJ_V, CTRL_HZ)


def test_stand_pose_matches_the_policy_default():
    """Tricks must return to the exact pose the RL policy's PD targets,
    otherwise handing control back to the walker jerks the robot."""
    np.testing.assert_allclose(tricks.STAND, DEFAULT_POSE)


@pytest.mark.parametrize("name", sorted(tricks.TRICKS))
def test_trick_compiles_to_valid_targets(name):
    seq = tricks.compile_trick(name)
    assert seq.ndim == 2 and seq.shape[1] == N_JOINTS
    assert len(seq) >= CTRL_HZ // 2, "trick is suspiciously short"
    assert np.isfinite(seq).all()
    np.testing.assert_allclose(seq[-1], tricks.STAND, atol=1e-6,
                               err_msg=f"{name} must end at the standing pose")


@pytest.mark.parametrize("name", sorted(tricks.TRICKS))
def test_trick_targets_respect_joint_limits(name, env):
    """Targets outside the XML joint ranges make the PD fight the limit."""
    seq = tricks.compile_trick(name)
    lo = env.model.jnt_range[1:N_JOINTS + 1, 0]
    hi = env.model.jnt_range[1:N_JOINTS + 1, 1]
    assert (seq >= lo - 1e-6).all(), f"{name} commands a joint below its limit"
    assert (seq <= hi + 1e-6).all(), f"{name} commands a joint above its limit"


@pytest.mark.parametrize("name", sorted(tricks.TRICKS))
def test_trick_keeps_the_robot_upright(name, env):
    """Replay the trick in physics; the robot must never tip and must recover
    to a normal standing height afterwards."""
    key = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(env.model, env.data, key)
    mujoco.mj_forward(env.model, env.data)

    seq = tricks.compile_trick(name, start_pose=env.data.qpos[N_FJ_Q:].copy())
    substeps = max(1, round((1.0 / CTRL_HZ) / env.model.opt.timestep))
    min_up = 1.0

    def pd(target):
        q, qd = env.data.qpos[N_FJ_Q:], env.data.qvel[N_FJ_V:]
        env.data.ctrl[:] = np.clip(KP * (target - q) - KD * qd,
                                   -TORQUE_LIMIT, TORQUE_LIMIT)
        mujoco.mj_step(env.model, env.data)

    for target in seq:
        for _ in range(substeps):
            pd(target)
        qx, qy = env.data.qpos[4], env.data.qpos[5]
        min_up = min(min_up, 1.0 - 2.0 * float(qx * qx + qy * qy))

    for _ in range(CTRL_HZ * substeps):        # 1 s to settle
        pd(tricks.STAND)

    assert min_up > 0.30, f"{name} tipped the robot over (uprightness {min_up:.2f})"
    z = float(env.data.qpos[2])
    assert z > 0.22, f"{name} left the robot collapsed at {z:.3f} m"


def test_unknown_trick_is_rejected():
    with pytest.raises(KeyError):
        tricks.compile_trick("backflip")


def test_available_lists_every_trick():
    assert set(tricks.available()) == set(tricks.TRICKS)
    for essential in ("sit", "wave", "dance", "bow", "lie"):
        assert essential in tricks.available()
