from legged_gym.envs.go2.go2_config_baseline import Go2BaseCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class Go2BackflipCfg(Go2BaseCfg):
    """Genesis-backflip compatible task configuration for Unitree Go2."""

    class env(Go2BaseCfg.env):
        num_envs = 4096
        num_observations = 60
        # 64 state features + 101 simulator-only dynamics/randomization values.
        num_privileged_obs = 165
        num_histroy_obs = 1
        num_env_priv_obs = 0
        train_type = "backflip"
        # The first 2 s contain the phase-conditioned flip. The final second
        # trains continuous RL landing recovery instead of resetting at phase end.
        episode_length_s = 3.0
        send_timeouts = True

    class terrain(Go2BaseCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        jump = False

    class commands(Go2BaseCfg.commands):
        curriculum = False
        resampling_time = 4.0

        class ranges(Go2BaseCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
            height_z = [0.0, 0.0]
            z_jump = [0.0, 0.0]
            z_normal = [0.0, 0.0]

    class init_state(Go2BaseCfg.init_state):
        pos = [0.0, 0.0, 0.32]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
            "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
        }

    class control(Go2BaseCfg.control):
        control_type = "P"
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.0}
        action_scale = 0.5
        decimation = 4
        use_actuator_network = False
        # Per-environment, per-episode random action delay at 50 Hz:
        # 0/1/2 policy steps = 0/20/40 ms.
        max_delay_steps = 2
        # Sensor-only observation delay at 50 Hz: 0/20/40 ms per episode.
        max_observation_delay_steps = 2
        # Official Unitree Go2HV actuator envelope used by unitree_rl_lab.
        # Full torque is available through X1; it then falls linearly to zero
        # at X2. Y1 is the drive limit and Y2 is the braking limit.
        motor_velocity_x1 = 13.5
        motor_velocity_x2 = 30.0
        motor_torque_y1 = 20.2
        motor_torque_y2 = 23.4
        joint_velocity_limits = [30.0] * 12
        target_velocity_limit = 13.5
        soft_velocity_limit = 24.0
        velocity_termination_start_ratio = 3.0
        velocity_termination_ratio = 1.05

    class asset(Go2BaseCfg.asset):
        # Feet are intentionally excluded. Head links are kept separate in the
        # Go2 URDF and must be penalized explicitly.
        penalize_contacts_on = ["base", "Head", "radar", "hip", "thigh", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 0

    class domain_rand(Go2BaseCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.0]
        randomize_center = True
        added_center_range = [-0.01, 0.01]
        # Implemented explicitly in Go2Backflip._compute_torques so it changes
        # both commanded torque and the effective saturation limit.
        randomize_motor_strength = False
        randomize_torque_scale = True
        torque_scale_range = [0.80, 1.00]
        # Per-motor no-load speed variation used by the torque-speed envelope.
        randomize_motor_velocity = True
        motor_velocity_scale_range = [0.85, 1.00]
        randomize_limb_mass = True
        limb_mass_scale_range = [0.95, 1.05]
        randomize_limb_inertia = True
        limb_inertia_scale_range = [0.90, 1.10]
        # Preserve the explicitly randomized inertia tensor instead of asking
        # PhysX to recompute it from collision geometry.
        recompute_inertia = False
        randomize_contact = True
        restitution_range = [0.0, 0.05]
        contact_offset_range = [0.0075, 0.0125]
        rest_offset_range = [-0.001, 0.001]
        randomize_joint_friction = False
        randomize_lag_timesteps = False
        randomize_Motor_Offset = True
        added_Motor_OffsetRange = [-0.02, 0.02]
        randomize_kp_scale = True
        kp_scale_range = [0.8, 1.2]
        randomize_kd_scale = True
        kd_scale_range = [0.8, 1.2]
        randomize_has_jumped = False
        push_robots = False

    class rewards(Go2BaseCfg.rewards):
        only_positive_rewards = False
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.80
        target_height = 0.30
        takeoff_start = 0.50
        takeoff_end = 0.75
        rotation_end = 1.00
        landing_start = 1.40
        max_pitch_rate = 7.2
        max_upward_velocity = 3.0
        phase_duration = 2.0
        flip_angle_rate_clip = 20.0
        flip_completion_angle = 5.50
        flip_success_angle = 5.80
        recovery_upright_cos = 0.90
        recovery_min_height = 0.24
        recovery_max_pitch_rate = 2.0
        recovery_pose_error = 0.30
        recovery_contact_force = 5.0
        body_contact_force = 100.0
        max_body_contact_penalty = 5.0
        min_head_center_height = 0.08
        head_contact_force = 40.0
        landing_force_threshold = 350.0
        max_landing_impact_penalty = 5.0
        # 24 environment steps are collected per PPO iteration. Safety terms
        # stay light for 250 iterations, then ramp to full strength by 1500.
        safety_curriculum_start = 0.05
        safety_curriculum_warmup_steps = 6000
        safety_curriculum_ramp_steps = 30000

        class scales:
            ang_vel_y = 5.0
            ang_vel_z = -1.0
            lin_vel_z = 20.0
            orientation_control = -1.0
            feet_height_before_backflip = -30.0
            height_control = -10.0
            default_pose = -5.0
            actions_symmetry = -0.1
            gravity_y = -10.0
            feet_distance = -1.0
            action_rate = -0.01
            action_jerk = -0.02
            dof_vel_limits = -5.0
            # Event rewards make a complete rotation and controlled landing
            # materially better than a high-speed half flip.
            rotation_progress = 50.0
            flip_completion = 500.0
            flip_success = 500.0
            recovery_upright = 5.0
            recovery_height = 3.0
            recovery_default_pose = 2.0
            recovery_still = 2.0
            recovery_feet_contact = 2.0
            # Motion-quality regularizers. They share the safety curriculum so
            # the policy first rediscovers the v2 flip and only then learns to
            # remove rear-leg flailing and post-landing oscillation.
            rear_leg_action_rate = -0.03
            rear_leg_symmetry = -0.5
            recovery_dof_velocity = -0.5
            # Only these four safety terms are added to the v2 reward.
            head_clearance = -5.0
            undesired_body_contact = -10.0
            dof_pos_limits = -5.0
            joint_target_limits = -0.5
            head_contact = -20.0
            landing_impact = -3.0

    class noise(Go2BaseCfg.noise):
        add_noise = True
        ang_vel_bias_range = [-0.05, 0.05]
        gravity_bias_range = [-0.01, 0.01]
        dof_pos_bias_range = [-0.01, 0.01]
        dof_vel_bias_range = [-0.20, 0.20]

        class noise_scales(Go2BaseCfg.noise.noise_scales):
            # Values below are in scaled actor-observation units.
            ang_vel = 0.025
            gravity = 0.01
            dof_pos = 0.01
            # Added after the 0.05 observation scaling: this corresponds to
            # at most about 1 rad/s of raw encoder-velocity noise.
            dof_vel = 0.025


class Go2BackflipCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        clip_param = 0.2
        desired_kl = 0.01
        entropy_coef = 0.005
        min_action_std = 0.35
        max_action_std = 1.50
        gamma = 0.99
        lam = 0.95
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4
        schedule = "adaptive"
        use_clipped_value_loss = True
        value_loss_coef = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 100
        experiment_name = "go2_backflip"
        run_name = ""
        resume = False
