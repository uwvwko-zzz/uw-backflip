from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2BaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # num_envs = 4096
        num_envs = 4096  # was getting a seg fault
        # num_envs = 100  # was getting a seg fault
        num_actions = 12
        num_observations = 45+1  #+3 for Standard algorithm # 45 only consider 3-dim commands #+ 10 # 10 more dims on command 
        # num_proprio_obs = 48
        camera_res = [1280, 720]
        camera_type = "d"  # rgb
        num_privileged_obs = 0 #200 + 5 +12 + 3 + 2 - 4 - 8#-4-2# 187, 5 means 4 mass and 1 z height and 3 world_ang_vel， 2 for XY position tracking
        train_type = "GenHis" #"GenHis"#"EST"  # standard, priv, lbc, standard, RMA, EST, Dream, GenHis

        follow_cam = False
        float_cam = False

        measure_obs_heights = False
        num_env_priv_obs = 17  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_histroy_obs = 5
        pass_has_jumped = True

        save_action = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'#'stair'#'stone'#'QRC'#
        jump = True
        origin_zero_z = True
        vis_type = 'train' #'test'

    class commands(LeggedRobotCfg.commands):
        heading_command = False           # 直接用 ang_vel_yaw 指令，让 Q/E 转向生效
        class ranges:
            lin_vel_x = [-1.0, 1.0]    # 对称，支持向后跳
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.8, 0.8]
            height_z = [0.32, 0.85]
            heading = [-1., 1.]
            vel_z_bool = [0, 1]
            z_jump = [0.55, 0.75]
            z_normal = [0.20, 0.45]

    class init_state(LeggedRobotCfg.init_state):
        # reference pose for passing QRC while walking
        # rel_foot_pos = [[0.220,0.220,-0.183,-0.183], # x
        #                 [0.156,-0.156,0.156,-0.156], # y
        #                 [-0.315,-0.315,-0.316,-0.316]]# z  # relative to the COM pos [FL FR RL RR]
        rel_foot_pos = [[0.194,0.194,-0.193,-0.193], # x
                        [0.156,-0.156,0.156,-0.156], # y
                        [-0.316,-0.316,-0.316,-0.316]]        
        rel_foot_pos_peak = [[0.232,0.232,-0.155,-0.155], # x
                        [0.148,-0.148,0.148,-0.148], # y
                        [-0.112,-0.112,-0.115,-0.115]] # z  # relative to the COM pos 
        pos = [0.1, 0.0, 0.34]  # x,y,z [m] # aliengo 0.39
        default_joint_angles = { # = target angles [rad] when action = 0.0

            'FL_hip_joint': 0.04,   # [rad]
            'RL_hip_joint': 0.04,   # [rad]
            'FR_hip_joint': -0.04,  # [rad]
            'RR_hip_joint': -0.04,   # [rad]

            'FL_thigh_joint': 0.722,#0.64,     # [rad]
            'RL_thigh_joint': 0.722,#0.69,   # [rad]
            'FR_thigh_joint': 0.722,#0.64,     # [rad]
            'RR_thigh_joint': 0.722,#0.69,   # [rad]

            'FL_calf_joint': -1.44,   # [rad]
            'RL_calf_joint': -1.44,    # [rad]
            'FR_calf_joint': -1.44,  # [rad]
            'RR_calf_joint': -1.44,    # [rad]
        }
        default_joint_angles_peak = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.04,  # [rad]
            "RL_hip_joint": 0.04,  # [rad]
            "FR_hip_joint": -0.04,  # [rad]
            "RR_hip_joint": -0.04,  # [rad]

            "FL_thigh_joint": 1.09,  # [rad]
            "RL_thigh_joint": 1.09,  # [rad]
            "FR_thigh_joint": 1.09,  # [rad]
            "RR_thigh_joint": 1.09,  # [rad]

            "FL_calf_joint": -2.25,  # [rad]
            "RL_calf_joint": -2.25,  # [rad]
            "FR_calf_joint": -2.25,  # [rad]
            "RR_calf_joint": -2.25,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "POSE"#"actuator_net2"#"actuator_net1" #"P"#"actuator_net"#"POSE" #
        # stiffness = {'joint': 20.}  # [N*m/rad]
        stiffness = {"joint": 40.0}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        damping = {"joint": 1.2}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # delay buffer:
        max_delay_steps = 5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        #actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go2_1st_f200_it6600_ly2_mlp_hist4.pt"
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/model_150.pth"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_description.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base", "trunk", "hip", "thigh"]#["base", "trunk", "hip","thigh"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]
        randomize_joint_friction = True
        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_center = True
        center_range = [-0.05, 0.05]

        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]

        randomize_lag_timesteps = True   # actuator net: True
        added_lag_timesteps = 4

        randomize_Motor_Offset = True  # actuator net: True
        added_Motor_OffsetRange = [-0.02, 0.02]

        randomize_has_jumped = True # Randomize if the robot has jumped or not at the start of the episode
        has_jumped_random_prob = 0.8
        reset_has_jumped = True # Whether to reset the has_jumped to False at a random point of the episode
        manual_has_jumped_reset_time = 0 # Manual time at which to reset has_jumped (in steps)

        class ranges():
            joint_friction_range = [0.0, 0.04]            

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.32
        max_contact_force = 1000.0
        only_positive_rewards = False #if walking policy: True, jumping policy: False 
        foot_height_target = 0.09
        max_height_reward_sigma = 0.05


        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 3.5#2 just for walking 1.5#3.0#2.0 #3.0#2.0#1.5 #3.5#3.0#1.2#2.0#1.0#0.6#1.5#1.0
            tracking_ang_vel = 1.5#1.0 just for walking 0.6#0.8#1.0#1.2#1.0#0.8 #1.5#1.75#1.0#0.8#0.6
            tracking_pitch_vel = 2.#4.0
            tracking_yaw = 0.6#0.7#0.6
            tracking_pitch = 0.
            lin_vel_z = 0.#1.5#3.5#-1.0 #-0.5 # base_lin_vel_z
            ang_vel_xy = 0.#-0.6#-0.6#-1.0#-0.6 # penalize on yaw
            orientation = -0.4#-1.0#-0.8#-0.6#-0.5#-0.5#0.2 positive means encourage the robot to stand upright
            upright = 0.#-0.2 # negative means encourage the robot to stand upright
            vel_switch = 0.#1.0
            tracking_pos = 0#1.5
            tracking_feet_pos = 0.#0.8

            torque_limits = 0.#-0.01 #0.            
            torques = -1e-6
            hip_torques = 0.#0.0001
            thigh_torques = 0.#0.00001 #0.0001
            calf_torques = 0.#-0.000005#-0.00001
            dof_acc = -2.5e-7
            feet_air_time = 1.#1.0 #1.0 for walking policy 0.
            stand_still = -3.0
            

            feet_distance = 0.#0.8#0.2#0.8#1.0#0.4#0.65
            feet_pos = 0.# 0 is for the walking policy 0.5 #0.4 #original design should be positive 0.6#0.6#0.8#0.4#0.6 # maybe need to be smaller
            early_contact = 0.#1.0
            height_track = 0.
            max_track = 0.#1.5#2.5
            task_max_height = 3.0  # raised from 2.0: push policy harder toward 0.68m target
                                   # (was 2.0; robot reaching ~0.54m in sim2sim, needs stronger signal)
            jumping = 30.#40.#1# 2 is for walking 30.0#40.#10.0 #25.0 #1.0

            collision = -1.0
            action_rate = -0.01 #-0.01
            # #### motion
            default_pose = 0.#-0.04#-0.08 0&-0.04 just for the walking policy#-0.12 #-0.1 #-0.12 # don't be too big!
            tracking_air_angle = 0.#-0.6#-0.6#-0.14#-0.1 #-0.5
            hip_motion = 0.#-0.06

            f_hip_motion = -0.02#0.#-0.2#-0.08
            r_hip_motion = -0.02#0.#-0.2#-0.08
            f_thigh_motion = -0.02#0.#-0.2#-0.04
            r_thigh_motion = -0.02#0.#-0.2#-0.04  
            f_calf_motion = -0.02#0.#-0.2#-0.04
            r_calf_motion = -0.02#0.#-0.2#-0.04
            dof_pos_limits = 0.#-10.

            flfr_gait_diff = 0.0#0.02#-0.02#-0.06#-0.2
            flfr_gait_diff2 = 0.  #-0.015
            rlrr_gait_diff = 0.0#0.02#-0.02#-0.06#-0.2
            rlrr_gait_diff2 = 0.#-0.015

            flrl_gait_diff = -0.0
            rlrr_gait_diff = -0.0

            flrl_gait_diff = -0.0#-0.2#-0.2#-0.2
            frrr_gait_diff = -0.0#-0.2#-0.2#-0.2

            # for troting gait:
            not_sync_contact = 0.#0.1#0.8 #0.08
            sync_contact = 0.#1. #0.08
            trot_contact = 0.#0.4 #0.4
            flrr_gait_diff = 0.#-0.02
            flrr_gait_diff2 = 0.#-0.015 #0.
            frrl_gait_diff = 0.#-0.02#-0.06#-0.2
            frrl_gait_diff2 = 0.#-0.015 #0.

            FL_phase_height = 0.#0.3
            FR_phase_height = 0.#0.3
            RL_phase_height = 0.#0.3
            RR_phase_height = 0.#0.3
            #### smoothness
            # dream_smoothness = -0.001
            # power_joint = -1e-4
            # foot_clearance = -0.001
            # foot_height = -0.01

            ##imitation reward
            imitation_root_pos = 0.#10.0
            imitation_joint_angle = 0.#-0.06
            imitation_feet_pos = 0.#-0.6#-1.0 #-0.8       

    class evals(LeggedRobotCfg.evals):
        feet_stumble = True
        feet_step = True
        crash_freq = True
        any_contacts = True

    class privInfo(LeggedRobotCfg.privInfo):
        enableMotorStrength = True
        enableMeasuredVel = True
        enableMeasuredHeight = True
        enableForce = True
        enable_priv_Zheights_weights = False #True
        enable_priv_feet_height = True
        enable_priv_ang_vel = True
        enable_priv_ZXYheights = True


class Go2BaseCfgPPO(LeggedRobotCfgPPO):

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 40000  # extended from 20000: resume from model_15000 for 25k more steps
        resume = False
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'go2'
        export_policy = False
        export_onnx_policy = False

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # Reduced vs base (1e-3) to stabilise training: action_std was diverging.
        # Adaptive KL schedule (desired_kl=0.01) will further adjust this.
        learning_rate = 5e-4
        # Reduced vs base (0.01): less exploration pressure so policy exploits
        # what it already knows about jumping rather than randomising again.
        entropy_coef = 0.005

    class Encoder(LeggedRobotCfgPPO.Encoder):
        priv_mlp_units = [258, 128, 8+12+3+2-4-8]# -4-2]#[258, 128, 3]  # 3 is for the vel estimator. 
        priv_info = False
        priv_info_dim = 200+5+12+3+2-4-8#-3 #-4-2# +2 for the XY position tracking: 197+3+10
        estLen = 3+1+4+12+3+2-4-8#-3 #-4-2# +2 for the XY position tracking
        proprio_adapt = False
        checkpoint_model = None
        proprio_adapt_out_dim = 11
        HistoryLen = 5
        Hist_info_dim = (45+1) * HistoryLen