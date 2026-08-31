from legged_gym.envs.go2_backflip.go2_backflip_config import (
    Go2BackflipCfg,
    Go2BackflipCfgPPO,
)


class DogBackflipCfg(Go2BackflipCfg):
    """Backflip configuration for the custom dog_1 URDF."""

    class env(Go2BackflipCfg.env):
        num_envs = 4096
        num_observations = 60
        num_privileged_obs = 165
        num_env_priv_obs = 0
        num_actions = 12

    class init_state(Go2BackflipCfg.init_state):
        pos = [0.0, 0.0, 0.344]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": -0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": 1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": 1.5,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.8,
            "RL_calf_joint": -1.5,
        }

    class control(Go2BackflipCfg.control):
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.0}
        action_scale = 0.5
        decimation = 4
        # URDF order: FL, FR, RR, RL; each leg is hip, thigh, calf.
        motor_torque_limits = [
            17.0, 17.0, 34.0, 17.0, 17.0, 34.0,
            17.0, 17.0, 34.0, 17.0, 17.0, 34.0,
        ]
        joint_velocity_limits = [
            30.1, 30.1, 20.07, 30.1, 30.1, 20.07,
            30.1, 30.1, 20.07, 30.1, 30.1, 20.07,
        ]
        target_velocity_limits = [13.5] * 12
        motor_velocity_x1 = [13.5] * 12
        soft_velocity_limits = [
            24.08, 24.08, 16.056, 24.08, 24.08, 16.056,
            24.08, 24.08, 16.056, 24.08, 24.08, 16.056,
        ]

    class asset(Go2BackflipCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog_1.urdf"
        name = "dog"
        foot_name = "foot"
        penalize_contacts_on = ["base", "hip", "thigh", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 1
        flip_visual_attachments = False
        collapse_fixed_joints = True

    class domain_rand(Go2BackflipCfg.domain_rand):
        nominal_base_mass = 13.555
        added_mass_range = [-1.0, 1.5]

    class rewards(Go2BackflipCfg.rewards):
        target_height = 0.344
        recovery_min_height = 0.27
        # No separate head link: base center is the head/trunk proxy.
        min_head_center_height = 0.10
        landing_force_threshold = 500.0


class DogBackflipCfgPPO(Go2BackflipCfgPPO):
    class runner(Go2BackflipCfgPPO.runner):
        experiment_name = "dog_backflip"
        run_name = ""
