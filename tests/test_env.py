"""Environment contract, invariants, and the properties that past bugs violated.

Each test here maps to a real defect found in this project's history:
  - heading invariance   -> the policy once only walked when facing east
  - PD holds the stance  -> raw torque control collapsed the robot to 0.15 m
  - obs/action contract  -> obs layout drifted from the config silently
"""

import numpy as np
import pytest
import mujoco

from sim.config import (OBS_DIM, ACT_DIM, N_JOINTS, N_FJ_Q, DEFAULT_POSE,
                        STAND_HEIGHT, MIN_Z, EnvCfg)
from sim.envs import RobotV2Env


# ------------------------------------------------------------------ contract
def test_obs_and_action_dims_match_config(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,), f"obs {obs.shape} != config OBS_DIM {OBS_DIM}"
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACT_DIM,)
    assert isinstance(info, dict)


def test_step_returns_valid_gym_tuple(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(obs).all(), "observation contains NaN/Inf"
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    for key in ("x_vel", "yaw_rate", "torso_z", "cmd_vx", "cmd_yaw"):
        assert key in info


def test_actions_are_clipped_not_rejected(env):
    """Out-of-range actions must be clipped, never crash or produce NaN."""
    env.reset(seed=0)
    obs, r, *_ = env.step(np.full(ACT_DIM, 50.0))
    assert np.isfinite(obs).all() and np.isfinite(r)


# --------------------------------------------------------------- determinism
def test_same_seed_gives_identical_rollout(env):
    def rollout():
        obs, _ = env.reset(seed=1234)
        traj = [obs]
        for i in range(20):
            obs, *_ = env.step(np.full(ACT_DIM, 0.1 * ((i % 3) - 1)))
            traj.append(obs)
        return np.array(traj)
    np.testing.assert_allclose(rollout(), rollout(), rtol=0, atol=0,
                               err_msg="same seed produced different rollouts")


# ---------------------------------------------------------- heading invariance
def test_observation_is_heading_invariant(env):
    """Rotating the robot about z must not change its observation.

    Regression guard: the first version rewarded WORLD-frame x-velocity, so the
    policy learned to walk only while facing east and froze facing west.
    """
    env.reset(seed=0)
    env.data.qpos[:] = env._stand_qpos
    mujoco.mj_forward(env.model, env.data)
    o1 = env._get_obs()

    for deg in (37, 90, 180, -113):
        a = np.radians(deg)
        env.data.qpos[3:7] = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
        mujoco.mj_forward(env.model, env.data)
        o2 = env._get_obs()
        assert np.abs(o1 - o2).max() < 1e-6, f"obs changed when facing {deg}°"


# ------------------------------------------------------------------- control
def test_pd_control_holds_standing_pose(env):
    """With action=0 the PD controller must hold the robot up.

    Regression guard: raw torque control let it collapse to ~0.15 m.
    """
    env.reset(seed=0, options={"command": [0.0, 0.0]})
    for _ in range(150):     # 3 s
        env.step(np.zeros(ACT_DIM))
    z = float(env.data.qpos[2])
    assert z > 0.22, f"robot sagged to {z:.3f} m holding the standing pose"
    joints = env.data.qpos[N_FJ_Q:]
    assert np.abs(joints - DEFAULT_POSE).max() < 0.25, "joints drifted off target"


def test_termination_on_fall(env):
    env.reset(seed=0)
    env.data.qpos[2] = MIN_Z - 0.02
    mujoco.mj_forward(env.model, env.data)
    assert env._is_terminated(), "a collapsed robot must terminate the episode"


def test_episode_truncates_at_max_steps():
    cfg = EnvCfg(); cfg.max_steps = 5
    cfg.domain_rand.enabled = False; cfg.obs_noise.enabled = False
    e = RobotV2Env(cfg=cfg)
    e.reset(seed=0)
    flags = [e.step(np.zeros(ACT_DIM))[3] for _ in range(5)]
    assert flags[-1] is True and not any(flags[:-1])
    e.close()


# ------------------------------------------------------- domain randomization
def test_domain_randomization_varies_dynamics(random_env):
    """DR must actually perturb the physics, and stay in the configured range."""
    seen_mass, seen_kp = set(), set()
    for s in range(12):
        random_env.reset(seed=s)
        seen_mass.add(round(float(random_env.model.body_mass[random_env._torso_bid]), 6))
        seen_kp.add(round(random_env._kp, 6))
    assert len(seen_mass) > 1, "torso mass never varied — DR is not applied"
    assert len(seen_kp) > 1, "PD gain never varied — DR is not applied"

    dr = random_env.cfg.domain_rand
    from sim.config import KP
    assert all(KP * dr.kp_scale[0] - 1e-9 <= k <= KP * dr.kp_scale[1] + 1e-9
               for k in seen_kp), "randomized gain escaped its configured range"


def test_randomization_does_not_compound_across_resets(random_env):
    """Params must be sampled from NOMINAL each episode, never stacked."""
    masses = []
    for s in range(30):
        random_env.reset(seed=s)
        masses.append(float(random_env.model.body_mass[random_env._torso_bid]))
    dr = random_env.cfg.domain_rand
    lo = random_env._nominal_mass + dr.torso_mass_kg[0] - 1e-6
    hi = random_env._nominal_mass + dr.torso_mass_kg[1] + 1e-6
    assert min(masses) >= lo and max(masses) <= hi, "mass drifted outside range"


def test_disabling_randomization_restores_nominal(env):
    env.reset(seed=0)
    from sim.config import KP, KD
    assert env._kp == KP and env._kd == KD
    assert float(env.model.body_mass[env._torso_bid]) == env._nominal_mass


# ------------------------------------------------------------ sensor noise
def test_obs_noise_perturbs_observation_but_not_state(random_env):
    """Noise must affect what the POLICY sees, never the physics or rewards."""
    random_env.reset(seed=3)
    mujoco.mj_forward(random_env.model, random_env.data)
    qpos_before = random_env.data.qpos.copy()
    o1 = random_env._get_obs()
    o2 = random_env._get_obs()
    assert not np.allclose(o1, o2), "noise enabled but observations are identical"
    np.testing.assert_array_equal(random_env.data.qpos, qpos_before,
                                  "reading obs must not mutate simulator state")


def test_reward_uses_true_state_not_noisy_obs(random_env):
    """Reward must be repeatable for a fixed state even with noise on."""
    random_env.reset(seed=5)
    a = np.zeros(ACT_DIM)
    r1 = random_env._compute_reward(a)
    r2 = random_env._compute_reward(a)
    assert abs(r1 - r2) < 1e-9, "reward is being polluted by observation noise"


# ------------------------------------------------------------------- reward
def test_reward_prefers_obeying_the_command(env):
    """Moving toward the commanded velocity must score better than ignoring it."""
    env.reset(seed=0, options={"command": [0.5, 0.0]})
    mujoco.mj_forward(env.model, env.data)
    a = np.zeros(ACT_DIM)
    env.data.qvel[0] = 0.0
    still = env._compute_reward(a)
    env.data.qvel[0] = 0.5
    moving = env._compute_reward(a)
    assert moving > still, "no reward gradient toward the commanded velocity"


def test_standing_command_penalises_drifting(env):
    env.reset(seed=0, options={"command": [0.0, 0.0]})
    mujoco.mj_forward(env.model, env.data)
    a = np.zeros(ACT_DIM)
    env.data.qvel[0] = 0.0
    still = env._compute_reward(a)
    env.data.qvel[0] = 0.6
    drifting = env._compute_reward(a)
    assert still > drifting, "told to stand still, drifting is not penalised"


# ------------------------------------------------------------ DR curriculum
def test_curriculum_scale_zero_collapses_ranges_to_nominal(random_env):
    """At scale 0 the robot must see exactly nominal dynamics, so early
    training is not handicapped by randomization it cannot yet cope with."""
    random_env.set_curriculum(0.0)
    for _ in range(8):
        random_env.reset()
        from sim.config import KP, KD
        assert abs(random_env._kp - KP) < 1e-9
        assert abs(random_env._kd - KD) < 1e-9
        assert abs(float(random_env.model.body_mass[random_env._torso_bid])
                   - random_env._nominal_mass) < 1e-9
    random_env.set_curriculum(1.0)


def test_curriculum_scale_one_restores_full_ranges(random_env):
    random_env.set_curriculum(1.0)
    dr = random_env.cfg.domain_rand
    assert dr.range_for("friction") == pytest.approx(dr.friction)
    assert dr.range_for("kp_scale") == pytest.approx(dr.kp_scale)


def test_curriculum_scale_interpolates_monotonically(random_env):
    dr = random_env.cfg.domain_rand
    widths = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        dr.scale = s
        lo, hi = dr.range_for("friction")
        widths.append(hi - lo)
    assert widths == sorted(widths), "randomization must widen monotonically"
    assert widths[0] == pytest.approx(0.0)
    dr.scale = 1.0


def test_curriculum_scale_is_clamped(random_env):
    dr = random_env.cfg.domain_rand
    dr.scale = 5.0
    assert dr.range_for("friction") == pytest.approx(dr.friction), "scale >1 must clamp"
    dr.scale = -3.0
    lo, hi = dr.range_for("friction")
    assert hi - lo == pytest.approx(0.0), "negative scale must clamp to nominal"
    dr.scale = 1.0


def test_curriculum_scales_observation_noise(random_env):
    """At scale 0 observations must be noiseless even with noise enabled."""
    random_env.set_curriculum(0.0)
    random_env.reset(seed=11)
    o1, o2 = random_env._get_obs(), random_env._get_obs()
    np.testing.assert_allclose(o1, o2, err_msg="noise present at curriculum scale 0")
    random_env.set_curriculum(1.0)
    o3, o4 = random_env._get_obs(), random_env._get_obs()
    assert not np.allclose(o3, o4), "noise missing at curriculum scale 1"
