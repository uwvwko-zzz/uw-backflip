# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 3072
        num_observations = 235
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_vel_obs  = None
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        train_type = "EST"  # standard, RMA, EST
        measure_obs_heights = False
        num_histroy_obs = 1
        num_env_priv_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        

    class terrain:
        mesh_type = 'plane'# 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        if mesh_type == 'QRC' or 'plane':
            obj_path = ['~/research/2024icraQRC/extreme-parkour/stepover_1.obj', '~/research/2024icraQRC/extreme-parkour/stepover_2.obj']        
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 0.2  # [m]
        curriculum = True   # curriculum training set to True, testing set to False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True


        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  #17 # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] #11
        # measured_points_x = [-2,-1.9,-1.8,-1.7,-1.6,-1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        #                      0.8,0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.7,1.8,1.9,2.0] # 41
        num_points = len(measured_points_x) * len(measured_points_y)
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 14.
        terrain_width = 14.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False#True
        num_goals = 8
        height = [0.02, 0.06]
        downsampled_scale = 0.075
        vis_type = 'train'

    class commands:
        curriculum = False #True
        max_curriculum = 1.
        num_commands = 6#5#4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        height_command = True
        heading_command = True  # if true: compute ang vel command from heading error
        tracking_z = True
        desired_jumping_height = 0.85

        jump_prob = 0.3
        zero_v_cmd_normal = False#True # if True, in normal mode, the commands are set to zero velocity  
        bool_jump = False#True # is true use the newcode structure

        class ranges:
            lin_vel_x = [-0.2, 1.0]#[1.0, 3.0] #[-0.6, 1.0] aliengo#[-0., 0.]#[-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5] #[-0., 0.] #[-0.5, 0.6] # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]#[-1., 1.] long_jump# [-1.5, 1.5] # [-0.00,0.00]#consider it as target_yaw #[-1, 1]  # min max [rad/s]\
            height_z = [0.32, 0.85] #0.85 #[0.45, 0.99]

            heading = [-1., 1.]#[-0.314,0.314]#[-3.14, 3.14]
            vel_z_bool = [0,1]

            z_jump = [0.55, 0.75] #root max height(command for jumping)
            z_normal = [0.20, 0.45] # command for walking #NOTE not used
 
        
    # class commands():
    #     jump_over_box = False
    #     num_commands = 13  # default: relative x,y,z for jump and desired quaternion (euler angles use xyz notation)
    #     # and 6 for centre of object and its dimensions.
    #     upward_jump_probability = 0.1
    #     curriculum = False
    #     curriculum_type = "time-based"
    #     randomize_commands = True
    #     curriculum_interval = 5
    #     max_curriculum = 1.

    #     num_levels = 11
    #     randomize_yaw = False
    #     resampling_time = 10.
        
    #     class ranges(): # command disturbance. Means the jumping distance range to track
    #         # The command distances are relative to the initial agent position and are sampled from
    #         # the ranges below:

    #         # This is the min/maximum ranges in the jump's distance curriculum (x_des = dx~pos_dx + x)
    #         pos_dx_lim = [-0.,0.] #pos_command_variation_limits
    #         pos_dy_lim = [-0.,0.]
    #         pos_dz_lim = [-0.0,0.0]
    #         # These are the starting ranges for the jump's distances (i.e. if curriculum 
    #         # if off, these stay the same for the whole training.)
    #         pos_dx_ini = [-0.,0.]  # pos_command_variation_ini
    #         pos_dy_ini = [-0.,0.]
    #         pos_dz_ini = [0.0,0.0]
    #         # These are the steps for the jump distance changes every curriculum update.
    #         pos_variation_increment = [0.01,0.01,0.01]

    #     class distances(): # Command distance to track. If you want to pass a fixed command distance, use these values:
    #         x = 0.
    #         y = 0.
    #         z = 0.
    #         # Specify in Euler 'xyz' notation (order is important as it gets converted to quat later):
    #         # des_angles_euler = [0.0,0.0,0.0]
    #         des_yaw = 0.6
            

    class init_state:
        pos = [0.0, 0.0, 0.41]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "joint_a": 0.,
            "joint_b": 0.}

    class control:
        control_type = 'POSE'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        actuator_net_file = None
    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        randomize_limb_mass = False
        added_limb_percentage = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

        randomize_center = False
        added_center_range = [-0.05, 0.05]

        randomize_motor_strength = False
        added_motor_strength = [1.0, 1.0]

        randomize_lag_timesteps = False
        added_lag_timesteps = 6

        randomize_Motor_Offset = False  # actuator net: True
        added_Motor_OffsetRange = [-0.02, 0.02]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.#1.0
            tracking_ang_vel = 0.#0.5
            lin_vel_z = 0.#-1.0
            ang_vel_xy = 0.#-0.05
            orientation = -0.0
            dof_vel = -0.0
            dof_acc = -2.5e-7
            action_rate = -0.01

            torques = 0
            base_height = -0.0
            feet_clearance = 0.#0.65
            feet_air_time = 0.0
            feet_air_time_base = 0.0
            feet_obs_contact = 0.0
            feet_step = -0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.0

            ### motion
            motion_base = 0.0
            motion = 0.0
            f_hip_motion = 0.0
            r_hip_motion = 0.0
            f_thigh_motion = 0.0
            r_thigh_motion = 0.0
            f_calf_motion = 0.0
            r_calf_motion = 0.0

            ###RMA
            RMA_work = 0.0
            RMA_ground_impact = 0.0
            RMA_smoothness = 0.0
            RMA_foot_slip = 0.0
            RMA_action_magnitude = 0.0

            ### dream
            orientation_base = 0.0
            dream_smoothness = 0.0
            power_joint = 0.0
            power_distribution = 0.0
            foot_clearance = 0.0
            foot_height = 0.0

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.0  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.9
        base_height_target = 0.9
        foot_height_target = 1.0
        max_contact_force = 100.  # forces above this value are penalized
    class evals:
        feet_stumble = False
        feet_step = False
        crash_freq = False
        any_contacts = False
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = False #True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.#0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-3, -3, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005 # 0.005 * 4 = 0.02 50Hz
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        enable_debug_viz = False#True #False

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class randomization:
        # Randomization Property
        randomizeMotorStrength = False
        randomizeMotorStrengthLower = 0.9
        randomizeMotorStrengthUpper = 1.1
        jointNoiseScale = 0.02

    class privInfo:
        enableMotorStrength = False
        enableMeasuredHeight = False
        enableMeasuredVel = False
        enableForce = False



class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 3000  # number of policy updates

        # logging
        save_interval = 250  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = True#False
        load_run = 'pose_normal1'  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        eval_baseline = False
        num_test_envs = 50
        export_policy = False
        export_onnx_policy = False
    class Encoder:
        priv_mlp_units = [256, 128, 8]
        decoder_mlp_units = [64, 128, 48]
        priv_info = False
        priv_info_dim = 17
        proprio_adapt = False
        checkpoint_model = None
        HistoryLen = None
        velLen = None
        Hist_info_dim = None