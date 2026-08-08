"""
Single source of truth for the robot's physical + control constants.

Before this module, PD gains and torque limits were duplicated in
`sim/envs/robot_v2_env.py` AND `brain/controller.py`, and the standing pose was
duplicated again in `brain/tricks.py`. Tuning one and not the others silently
made the trick player and the RL policy drive the robot with *different*
controllers — a latent bug that produces "it walks fine but tricks feel wrong".
Everything now imports from here.

Domain randomization and observation-noise settings live here too so training
runs are described by data rather than by scattered literals.
"""

from dataclasses import dataclass, field, asdict
import numpy as np

# --------------------------------------------------------------------- robot
N_JOINTS = 12          # 3 per leg (hip abduction, shoulder, knee) x 4 legs
N_FJ_Q   = 7           # free joint: xyz(3) + quat(4)
N_FJ_V   = 6           # free joint: linvel(3) + angvel(3)

# Standing pose per leg: [hip, shoulder, knee]. PD target when action == 0.
DEFAULT_POSE = np.array([0.0, 0.63, -1.25] * 4)
# Per-joint torque ceiling (N·m), same ordering.
TORQUE_LIMIT = np.array([8.0, 12.0, 14.0] * 4)

# ------------------------------------------------------------------- control
KP           = 25.0    # PD position gain
KD           = 0.7     # PD damping gain
ACTION_SCALE = 0.25    # rad — scales policy output to a joint-target offset
CTRL_DT      = 0.02    # 50 Hz control loop
CTRL_HZ      = int(round(1.0 / CTRL_DT))

STAND_HEIGHT = 0.27    # target torso height (m)
MIN_Z        = 0.14    # below this the episode ends (fallen)
UPRIGHT_MIN  = 0.5     # projected-gravity z cutoff for "tipped over"

# --------------------------------------------------------------- gait clock
GAIT_FREQ_MIN = 2.4    # Hz at a standstill
GAIT_FREQ_MAX = 4.6    # Hz at top commanded speed
GAIT_FREQ_REF = 0.6    # speed (m/s) at which cadence saturates
TROT_PAIR_A   = (0, 3) # FL + RR form one diagonal pair

# ------------------------------------------------------------------ commands
VX_RANGE  = (-0.3, 0.6)   # m/s   (backward .. forward)
YAW_RANGE = (-0.8, 0.8)   # rad/s (right .. left)


@dataclass
class DomainRand:
    """Dynamics randomization, resampled every episode.

    Standard practice for legged RL — a policy trained on one exact set of
    physical parameters overfits to them and degrades under any mismatch
    (different friction, payload, weaker motors, sensor lag). See legged_gym /
    walk-these-ways, which randomize friction and mass for sim-to-real.

    Ranges are multiplicative unless noted.
    """
    enabled: bool = True
    friction: tuple = (0.6, 1.4)        # x nominal foot/ground friction
    torso_mass_kg: tuple = (-0.15, 0.30)  # ADDITIVE payload on the torso (kg)
    kp_scale: tuple = (0.85, 1.15)      # x PD position gain (motor variation)
    kd_scale: tuple = (0.85, 1.15)      # x PD damping gain
    torque_scale: tuple = (0.85, 1.10)  # x torque ceiling (battery sag, wear)


@dataclass
class ObsNoise:
    """Per-step Gaussian sensor noise (std, in each signal's own units).

    The robot's observation comes from an IMU + joint encoders on real
    hardware; training on a perfect noiseless state produces a policy that is
    brittle to the jitter any real sensor has. Zero-mean noise at realistic
    magnitudes is the cheapest robustness win available.
    """
    enabled: bool = True
    height: float     = 0.005   # m
    gravity: float    = 0.02    # unit vector components
    lin_vel: float    = 0.05    # m/s
    ang_vel: float    = 0.10    # rad/s
    joint_pos: float  = 0.01    # rad
    joint_vel: float  = 0.60    # rad/s (encoder differentiation is noisy)


@dataclass
class RewardCfg:
    """Reward term weights. Grouped so a training run is fully described by
    this object plus the seed (see training/run_manifest.py)."""
    track_lin_progress: float = 5.0
    track_lin_exp:      float = 1.5
    track_ang_progress: float = 3.0
    track_ang_exp:      float = 1.0
    height:             float = 1.5
    upright:            float = 1.0
    alive:              float = 0.3
    gait_match:         float = 0.5
    swing_clearance:    float = 6.0
    foot_slip:          float = 2.0
    lateral:            float = 0.3
    yaw_drift:          float = 1.0
    stillness:          float = 3.0
    roll_pitch:         float = 0.3
    action_rate:        float = 0.05
    joint_vel:          float = 0.0002
    foot_lift:          float = 1.5
    max_foot_lift:      float = 0.12   # m — swing clearance cap
    target_swing_clear: float = 0.07   # m — rewarded swing height


@dataclass
class EnvCfg:
    """Everything that defines the training environment."""
    domain_rand: DomainRand = field(default_factory=DomainRand)
    obs_noise:   ObsNoise   = field(default_factory=ObsNoise)
    reward:      RewardCfg  = field(default_factory=RewardCfg)
    max_steps:   int        = 1000
    stand_prob:  float      = 0.12

    def to_dict(self):
        return asdict(self)


# Observation layout — keep in sync with RobotV2Env._get_obs().
OBS_DIM = (1        # torso height
           + 3      # projected gravity (body frame)
           + 3      # body linear velocity
           + 3      # body angular velocity
           + N_JOINTS   # joint pos - default
           + N_JOINTS   # joint vel
           + N_JOINTS   # last action
           + 2      # gait clock (sin, cos)
           + 2)     # command (vx, yaw)
ACT_DIM = N_JOINTS
