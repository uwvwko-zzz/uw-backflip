

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
from numpy.linalg import norm, inv
from math import acos, atan2, sqrt, pi
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import os
import time

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.lstm_latent_model import ACNet 
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.load_data import ref_data

def no_jump(func):
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs) * (self.commands_z.squeeze(1) <= 0.5)
    return wrapper

class Aliengo:
    dist_hip_x = 0.2399
    dist_hip_y = 0.051
    len_hip = 0.083
    len_thigh = 0.25
    len_calf = 0.25

class Go2:
    dist_hip_x = 0.1934
    dist_hip_y = 0.0465
    len_hip = 0.0955
    len_thigh = 0.213
    len_calf = 0.213

class IK:
    lf = np.array([1,1,1])
    rf = np.array([1,-1,1])
    lh = np.array([-1,1,1])
    rh = np.array([-1,-1,1])
    
    def __init__(self,robot,leg='LF'):
        
        self.dist_hip_x = robot.dist_hip_x
        self.dist_hip_y = robot.dist_hip_y
        self.len_hip    = robot.len_hip
        self.len_thigh  = robot.len_thigh
        self.len_calf   = robot.len_calf
        self.avoid_singular_param = 0.95
        self.ws_radius = (self.len_calf+self.len_thigh)*self.avoid_singular_param

        hip_xyz_B    = np.array([self.dist_hip_x, self.dist_hip_y, 0])
        point_in_hip_axis  = np.array([0.0, self.dist_hip_y, 0])
        thigh_xyz_B  = np.array([self.dist_hip_x ,self.dist_hip_y+self.len_hip, 0])
        calf_xyz_B   = np.array([self.dist_hip_x ,self.dist_hip_y+self.len_hip, -self.len_thigh])
        _foot_xyz_B  = np.array([self.dist_hip_x ,self.dist_hip_y+self.len_hip, -self.len_thigh-self.len_calf])
        
        if leg == 'LF':
            self.hip_xyz_B    = self.lf*hip_xyz_B
            self.point_in_hip_axis  = self.lf*point_in_hip_axis
            self.thigh_xyz_B  = self.lf*thigh_xyz_B
            self.calf_xyz_B   = self.lf*calf_xyz_B
            self._foot_xyz_B  = self.lf*_foot_xyz_B

        
        elif leg == 'RF':
            self.hip_xyz_B    = self.rf*hip_xyz_B
            self.point_in_hip_axis  = self.rf*point_in_hip_axis
            self.thigh_xyz_B  = self.rf*thigh_xyz_B
            self.calf_xyz_B   = self.rf*calf_xyz_B
            self._foot_xyz_B  = self.rf*_foot_xyz_B

        elif leg == 'LH':
            self.hip_xyz_B    = self.lh*hip_xyz_B
            self.point_in_hip_axis  = self.lh*point_in_hip_axis
            self.thigh_xyz_B  = self.lh*thigh_xyz_B
            self.calf_xyz_B   = self.lh*calf_xyz_B
            self._foot_xyz_B  = self.lh*_foot_xyz_B

        elif leg == 'RH':
            self.hip_xyz_B    = self.rh*hip_xyz_B
            self.point_in_hip_axis  = self.rh*point_in_hip_axis
            self.thigh_xyz_B  = self.rh*thigh_xyz_B
            self.calf_xyz_B   = self.rh*calf_xyz_B
            self._foot_xyz_B  = self.rh*_foot_xyz_B

        else:
            raise ValueError('leg should be LF, RF, LH, or RH')


    def compute_inverse(self,foot_xyz_B, foot_vel_B, hip_angle):
        #-------calculate calf using variant type of question 2

        foot2hip_B = foot_xyz_B - self.hip_xyz_B
        foot2hip_len = norm(foot2hip_B)
        if foot2hip_len> self.ws_radius:
            #print('Warning: foot position %s is out of workspace'%repr(foot_xyz_B.tolist()))
            foot_xyz_B = self.hip_xyz_B + foot2hip_B/foot2hip_len*self.ws_radius
            foot2hip_len = self.ws_radius
            #print('\t changed to %s\n'%repr(foot_xyz_B.tolist()))
        foot2thigh_len = sqrt(foot2hip_len*foot2hip_len - self.len_hip*self.len_hip)
        calf = acos((self.len_thigh*self.len_thigh+self.len_calf*self.len_calf-foot2thigh_len*foot2thigh_len)/(2*self.len_thigh*self.len_calf))
        calf = calf - pi # rerange by definition
        # print(calf)
        #-------calculate thigh
        foot2calf_B = np.array([-self.len_calf*np.sin(calf),0.0,-self.len_calf*np.cos(calf)]) # Ry(calf)*p
        q = self.point_in_hip_axis
        r = self.hip_xyz_B
        p = self.calf_xyz_B + foot2calf_B # foot in base frame with only rotated around calf
        v = q-r
        u = p-r
        w = np.array([0.0, 1.0, 0.0])
        u_ = u - w*np.dot(u,w)
        v_ = v - w*np.dot(v,w)
        delta = norm(foot_xyz_B-q)
        delta_ = sqrt(pow(delta,2) - pow(np.dot(w,p-q),2))

        
        theta_0  = atan2(np.dot(np.cross(u_,v_),w),np.dot(u_,v_))
        theta_0_ = acos((np.dot(u_,u_)+np.dot(v_,v_)-pow(delta_,2))/(2*norm(u_)*norm(v_))) # 0~pi

        #rerange by definition to keep thigh>0 so that only one solution
        if theta_0 > 0:
            thigh = theta_0 - theta_0_
        else:
            thigh = theta_0 + theta_0_
            
        # #----calculate hip using question 1
        rot = R.from_rotvec(thigh*np.array([0.0, 1.0, 0.0])).as_matrix()
        p = self.thigh_xyz_B + np.dot(rot, np.array([0.0,0.0,-self.len_thigh]) + foot2calf_B) # before rotation
        q = foot_xyz_B # after rotation
        r = self.hip_xyz_B
        w = np.array([1.0, 0.0, 0.0])
        u = p - r
        v = q - r
        u_ = u - w*np.dot(u,w)
        v_ = v - w*np.dot(v,w)
        hip = atan2(np.dot(np.cross(u_,v_),w),np.dot(u_,v_)) # define the hip as the default value
        hip = hip_angle
        
        #calculate body frame jacobian
        J_ = np.zeros((3,3))
        J_[:,0] = -np.cross(foot2hip_B, np.array([1.0, 0.0, 0.0]))
        rot_hip = R.from_rotvec(hip*np.array([1.0, 0.0, 0.0])).as_matrix()
        rot_thigh = R.from_rotvec(thigh*np.array([0.0, 1.0, 0.0])).as_matrix()
        foot2thigh_thigh = np.array([-self.len_calf*np.sin(calf),0.0,-self.len_calf*np.cos(calf)-self.len_thigh])
        J_[:,1] = -rot_hip@rot_thigh@np.cross(foot2thigh_thigh,np.array([0,1,0]))
        rot_calf = R.from_rotvec(calf*np.array([0.0, 1.0, 0.0])).as_matrix()
        foot2calf_calf = np.array([0,0,-self.len_calf])
        J_[:,2] = -rot_hip@rot_thigh@rot_calf@np.cross(foot2calf_calf,np.array([0,1,0]))
        
        joint_pos = np.array([hip,thigh,calf])
        joint_vel = np.linalg.lstsq(J_, foot_vel_B, rcond=None)[0]
        return joint_pos,joint_vel
    
    
class RobotIK:
    def __init__(self,robot):
        self.robot =robot
        self.ik = [IK(robot,'LF'), IK(robot,'RF'), IK(robot,'LH'), IK(robot,'RH')] # FL FR RL RR 0 1 2 3

    def computeIK(self, foot_pos, foot_vel):
        joint_pos = np.zeros(12)
        joint_vel = np.zeros(12)
        left_hip = 0.04 #go2:4, aliengo:6
        right_hip = -0.04
        for i in range(4):
            foot_pos_i = foot_pos[3*i:3*(i+1)]
            foot_vel_i = foot_vel[3*i:3*(i+1)]
            if i%2 == 0:
                joint_pos_i, joint_vel_i = self.ik[i].compute_inverse(foot_pos_i, foot_vel_i,left_hip)
            else:
                joint_pos_i, joint_vel_i = self.ik[i].compute_inverse(foot_pos_i, foot_vel_i,right_hip)
            joint_pos[3*i:3*(i+1)] = joint_pos_i  # hip, calf, thigh
            joint_vel[3*i:3*(i+1)] = joint_vel_i  # hip, calf, thigh
        return joint_pos, joint_vel

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training
        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.num_calls = 0
        self.cfg = cfg


        self._setup_priv_option_config(self.cfg.privInfo)


        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = self.cfg.sim.enable_debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)


        self.priv_info_dict = {
                'base_mass': (0, 1),
                'limb_mass': (1, 4),  # hip, thigh, calf
                'center': (4, 6),
                'motor_strength': (6, 18),  # weaken or strengthen
                'friction': (18, 19),
            }

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
    
        self.save_actions = cfg.env.save_action
        if self.save_actions:
            self.action_log = []
            self.start_time = time.time()
            self.log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'actions')
            os.makedirs(self.log_dir, exist_ok=True)

        # load actuator network
        if self.cfg.control.control_type == "actuator_net2":
            #print("********* Load actuator net from *********", self.cfg.control.actuator_net_file)
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            args = {}
            args['use_cuda'] = True
            args['encoder_size'] = 128
            args['decoder_size'] = 256
            args['in_length'] = 1
            args['out_length'] = 5
            args['grid_size'] = (13,3)
            args['input_embedding_size'] = 32
            args['joint_classes'] = 3
            args['use_maneuvers'] = True
            args['train_flag'] = False
            args['batch_size'] = self.num_envs #*12
            args['val_batch_size'] = self.num_envs #*12
            actuator_network = ACNet(args, 2)
            ckpt = torch.load(actuator_network_path)
            actuator_network.load_state_dict(ckpt['state_dict'])
            #print('loading model successfully')
            if args['use_cuda']:
                actuator_network = actuator_network.cuda()
            #actuator_network = torch.jit.load(actuator_network_path).to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_pos_last_4, joint_pos_last_5, joint_vel, joint_vel_last,
                                      joint_vel_last_last, joint_vel_last_4, joint_vel_last_5):
# self.dof_pos & self.dof_vel: (FL FR RL RR)*(hip, thigh, calf)
                xs_pos = torch.cat((joint_pos_last_5.unsqueeze(-1),
                                    joint_pos_last_4.unsqueeze(-1),
                                    joint_pos_last_last.unsqueeze(-1),
                                    joint_pos_last.unsqueeze(-1),
                                    joint_pos.unsqueeze(-1)), dim=-1) # 4096, 12, 5
                xs_vel = torch.cat((joint_vel_last_5.unsqueeze(-1),
                                    joint_vel_last_4.unsqueeze(-1),
                                    joint_vel_last_last.unsqueeze(-1),
                                    joint_vel_last.unsqueeze(-1),
                                    joint_vel.unsqueeze(-1)), dim=-1) # 4096, 12, 5
                xs = torch.cat((xs_pos.unsqueeze(-1), xs_vel.unsqueeze(-1)),dim=-1) # 4096,12,5,2  12:FL FR RL RR * (hip thigh calf)
                xs = xs.permute(1,0,2,3) # 12, 4096, 5, 2

                # build the joint labels
                hip_label = torch.tensor([[1,0,0]]).unsqueeze(0) # 1,1,3
                hip_labels = hip_label.repeat(4, self.num_envs, 1) #4,4096,3
                thigh_label = torch.tensor([[0,1,0]]).unsqueeze(0)
                thigh_labels = thigh_label.repeat(4, self.num_envs, 1)    
                calf_label = torch.tensor([[0,0,1]]).unsqueeze(0)
                calf_labels = calf_label.repeat(4, self.num_envs, 1)  

                # 写法1：
                torques = []
                for i in range(4):
                    hip_tau, _h = actuator_network(xs[3*i].cuda(), hip_labels[i,:,:].cuda())
                    thigh_tau, _t = actuator_network(xs[3*i+1].cuda(), thigh_labels[i,:,:].cuda())
                    calf_tau, _c = actuator_network(xs[3*i+2].cuda(), calf_labels[i,:,:].cuda()) # 4096, 5, 1
                    torques += [hip_tau[:,-1,:].data, thigh_tau[:,-1,:].data, calf_tau[:,-1,:].data]
                output_torques = torch.cat(torques,dim=1) # 4096, 12

                # 写法2： 速度跟1差不多，显存反而占用更高
                # actu_input = torch.cat((xs[0],xs[3],xs[6],xs[9]), dim=0)
                # actu_input = torch.cat((actu_input, xs[1],xs[4],xs[7],xs[10]), dim=0)
                # actu_input = torch.cat((actu_input, xs[2],xs[5],xs[8],xs[11]), dim=0) # (12*4096, 5, 2)
                # joint_labels = torch.cat((hip_labels, thigh_labels, calf_labels), dim=0) # (12*4096, 3)
                # full_tau, _f = actuator_network(actu_input.cuda(), joint_labels.cuda())  # full_tau is (12*4096, 5, 1) #12: 4hip, 4thigh, 4calf 
                # full_tau = full_tau[:,-1,:].data
                # output_temp = torch.split(full_tau,self.num_envs, dim=0) # each is (4096 ,1)
                # #print('output_temp is:',len(output_temp))
                # output_torques = torch.cat((output_temp[0],output_temp[4],output_temp[8],output_temp[1],output_temp[5],output_temp[9],output_temp[2],output_temp[6],output_temp[10],output_temp[3],output_temp[7],output_temp[11]), dim=1)
                # 

                return output_torques # 不能这么写，tensor 只要经过了reshape 或者permute 就不再continuous, 不可以view().
            
            self.actuator_network = eval_actuator_network

        elif self.cfg.control.control_type == "actuator_net1":
            #print("********* Load actuator net from *********", self.cfg.control.actuator_net_file)
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            actuator_network = torch.jit.load(actuator_network_path).to(self.device)

            # def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_pos_last_last_last, joint_vel, joint_vel_last,
            #                           joint_vel_last_last, joint_vel_last_last_last):
            #     xs = torch.cat((joint_pos.unsqueeze(-1),
            #                     joint_pos_last.unsqueeze(-1),
            #                     joint_pos_last_last.unsqueeze(-1),
            #                     joint_pos_last_last_last.unsqueeze(-1),
            #                     joint_vel.unsqueeze(-1),
            #                     joint_vel_last.unsqueeze(-1),
            #                     joint_vel_last_last.unsqueeze(-1),
            #                     joint_vel_last_last_last.unsqueeze(-1)), dim=-1)
            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6)) #6
                return torques.view(self.num_envs, 12)

            
            self.actuator_network = eval_actuator_network

    def make_handle_trans(self, res, env_num, trans, rot, hfov=None):
        # TODO Add camera sensors here?
        camera_props = gymapi.CameraProperties()
        # print("FOV: ", camera_props.horizontal_fov)
        # camera_props.horizontal_fov = 75.0
        # 1280 x 720
        width, height = res
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True
        if hfov is not None:
            camera_props.horizontal_fov = hfov
        # print("envs[i]", self.envs[i])
        # print("len envs: ", len(self.envs))
        camera_handle = self.gym.create_camera_sensor(
            self.envs[env_num], camera_props
        )
        # print("cam handle: ", camera_handle)

        local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(75.0, 75.0, 30.0)
        # local_transform.r = gymapi.Quat.from_euler_zyx(0, 3.14 / 2, 3.14)
        x, y, z = trans
        local_transform.p = gymapi.Vec3(x, y, z)
        a, b, c = rot
        local_transform.r = gymapi.Quat.from_euler_zyx(a, b, c)

        return camera_handle, local_transform

    def save_action(self, delayed_actions):
        if not self.save_actions:
            return

        # Convert to numpy array and move to CPU if necessary
        actions_np = delayed_actions.cpu().numpy()
        
        # Get current time step
        current_time = time.time() - self.start_time
        
        # Append to action log
        self.action_log.append((current_time, actions_np))

        # Periodically save to file (e.g., every 1000 steps)
        if len(self.action_log) >= 4000:
            self._save_actions_to_file()

    def _save_actions_to_file(self):
        if not self.action_log:
            return

        # Convert action log to structured numpy array
        dtype = [('time', float), ('actions', float, (self.num_envs, 12))]
        actions_array = np.array(self.action_log, dtype=dtype)
        
        # Generate filename with timestamp
        filename = f'actions_{time.strftime("%Y%m%d-%H%M%S")}.npy'
        filepath = os.path.join(self.log_dir, filename)
        
        # Save to file
        np.save(filepath, actions_array)
        #print(f"Saved actions to {filepath}")
        
        # Clear the action log
        self.action_log = []

    #def step(self, actions, estimations):
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.action_buffer = torch.roll(self.action_buffer, shifts=1, dims=0)
        self.action_buffer[0] = self.actions
        delayed_actions = self.action_buffer[self.env_delay_steps, torch.arange(self.num_envs, device=self.device)]
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):

            ### torques change 1 ############
            if self.cfg.control.control_type == "POSE":
                # Note: Position control
                self.target_poses = self._compute_poses(self.actions).view(self.target_poses.shape)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_poses))
                #self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            else:
                # Note: Torques control
                if self.cfg.env.save_action:
                    try:
                        self.save_action(delayed_actions)
                    except KeyboardInterrupt:
                        if self.cfg.env.save_action:
                            print("Interrupt received. Saving actions...")
                            self._save_actions_to_file()
                            raise  # Re-raise the exception after saving
                #self.torques = self._compute_torques(delayed_actions).view(self.torques.shape)
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # isaacGYM 仿真環境與gym學習環境交互。把算出來的torques 傳入。
                                                                                                        # gym_torch.unwrap_tensor is to wrap the input into a PyTorch Tensor object
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()#estimations)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        self.obs_dict['obs'] = self.obs_buf
        self.obs_dict['privileged_info'] = self.privileged_obs_buf.to(self.device)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.device)
        # self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device).flatten(1)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device).flatten(1)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):#, estimations):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg.control.control_type == "POSE":
            # Note: Position control
            self.gym.refresh_dof_force_tensor(self.sim)

        self.episode_length_buf += 1 # compute the steps in each episode.
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec) # 将gravity_vec 转到base_quat的坐标系下
        #print("----------------------------------------------")
        #print("feet indices are:", self.feet_indices)
        self.feet_pos = self.rigid_body_state[:,self.feet_indices,0:3]
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        #print("bool z vel command:", self.commands[:,4])

        self._post_physics_step_callback()

        #print("normal contact forces are: ", torch.sum(self.contact_forces[:, self.feet_indices, 2], dim=-1))
        foot_height  = self.rigid_body_state[:, self.feet_indices, 2:3]
        feet_pos = self.rigid_body_state[:, self.feet_indices, 0:3]
        foot_vel = self.rigid_body_state[:, self.feet_indices, 7:9]

        self.foot_height = torch.cat((foot_height[:, 0, ],
                                      foot_height[:, 1, ],
                                      foot_height[:, 2, ],
                                      foot_height[:, 3, ],
                                      ), dim=-1)
        self.FL_feet_pos = feet_pos[:,0,:]
        self.FR_feet_pos = feet_pos[:,1,:]
        self.RL_feet_pos = feet_pos[:,2,:]
        self.RR_feet_pos = feet_pos[:,3,:]
        
        # self.foot_vel = torch.cat((foot_vel[:, 0, ],
        #                            foot_vel[:, 1, ],
        #                            foot_vel[:, 2, ],
        #                            foot_vel[:, 3, ],
        #                               ), dim=-1)
        # print('foot', foot_height)
        # compute observations, rewards, resets, ...

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        contact2 = self.contact_forces[:, self.feet_indices]
        #if torch.all(contact, dim =-1):
        #print(contact2)
        feet_relative = self.feet_pos[:, :, :3] - self.root_states[:, :3].unsqueeze(1)
        feet_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        for i in range(4):
            feet_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, feet_relative[:,i,:]) 
        #print('torque', self.fsdata)
        #print('z_height', self.root_states[0,2].item())
        #print('contact', contact2.cpu().numpy())
        #print('torque', self.torques.cpu().numpy())
        #print('vel', torch.norm(self.root_states[0,7:10]).cpu().numpy())
        #print('feet_height', torch.mean(feet_body_frame[0,:,-1]).cpu().numpy())

        self.last_has_jumped = self.has_jumped
        self.check_jump()
        #print("---------self.has_jumped-------", self.has_jumped)
        self._store_states(self.mid_air)#(self.landing_ids)
        idx = self.mid_air * ~self.has_jumped * self.was_in_flight
        #self.max_height[idx] = torch.max(self.max_height[idx],self.root_states[idx, 2]) 
        # idx = torch.logical_and(self.mid_air,~self.has_jumped)
        self.task_max_height[idx] = torch.max(self.task_max_height[idx],self.root_states[idx, 2]) 
        self.max_height = self.compute_max_height()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.check_has_jump_reset()
        self.reset_idx(env_ids) # reset
        
        # if self.cfg.domain_rand.randomize_has_jumped and self.cfg.domain_rand.reset_has_jumped:
        #     rand_envs = self._has_jumped_rand_envs
        #     idx = torch.nonzero(torch.logical_and(rand_envs,self._reset_randomised_has_jumped_timer == self.episode_length_buf),as_tuple=False).flatten()
        #     if not self.cfg.domain_rand.manual_has_jumped_reset_time == 0:
        #         idx = torch.nonzero(torch.logical_and(rand_envs,self.cfg.domain_rand.manual_has_jumped_reset_time == self.episode_length_buf),as_tuple=False).flatten()
        #     self.has_jumped[idx] = False
        #     self._has_jumped_rand_envs[idx] = False
        #     self._reset_randomised_has_jumped_timer[idx] = 0
        #     # Keep track of when the has_jumped flag was switched back to 0 (for certain terminations):
        #     self._has_jumped_switched_time[idx] = self.episode_length_buf[idx]

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions_2[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contact_forces[:] = self.contact_forces[:]
        self.last_torques[:] =  self.torques[:]
        self.last_forward = self.forward

        # idx = self.mid_air * ~self.has_jumped * self.was_in_flight
        # # idx = torch.logical_and(self.mid_air,~self.has_jumped)
        # self.task_max_height[idx] = torch.max(self.task_max_height[idx],self.root_states[idx, 2]) 
        # self.max_height[self.mid_air] = torch.max(self.max_height[idx],self.root_states[idx, 2]) 
        # self.check_has_jump_reset()
        #print('mean height is among all agents:', torch.mean(self.max_height))   
        # #idx = self.mid_air 
        # #self.max_height[idx] = torch.max(self.max_height[idx],self.root_states[idx, 2]) # update max height robot can achieved so far
        
        # idx = self.mid_air * ~self.has_jumped * self.was_in_flight
        # # idx = torch.logical_and(self.mid_air,~self.has_jumped)
        # self.max_height[idx] = torch.max(self.max_height[idx],self.root_states[idx, 2]) # update max height robot can achieved so far
        
        # self.min_height[~self.has_jumped] = torch.min(self.min_height[~self.has_jumped], self.root_states[~self.has_jumped, 2]) # update min height achieved

        # # temp = torch.cat((self.max_height[:].view(1,-1),self.root_states[:, 2].view(1,-1)), dim=0)
        # # self.max_height[:] = torch.max(temp, dim=0).values
        #print("torch.cuda.memory_allocated: %fMB"%(torch.cuda.memory_allocated(0)/1024/1024))

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis(estimations)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device) 
        height_cutoff = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) < -0.5  # root_states[:,2].unsqueeze(1) is num_envs,1 # measred_heights is num_envs, 187
        #height_cutoff_up = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) > self.commands[:,3] + 0.04
        height_cutoff_up = self.root_states[:, 2] > (self.commands[:,3] + 0.06)
        #height_cutoff_low = torch.logical_and(torch.mean(self.task_max_height.unsqueeze(1) - self.measured_heights, dim=1) < self.commands[:,3] - 0.03, self.has_jumped)
        height_cutoff_low = torch.logical_and(self.task_max_height < (self.commands[:,3] - 0.05), self.has_jumped)  # task_max_height 會重置
        #height_cutoff_low = torch.logical_and(self.max_height < (self.commands[:,3] - 0.07), self.has_jumped) # max_height 不會重置
        #self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
        #                           dim=1)
        # bounding box added:
        rewed_ids = self.task_max_height>self.commands[:,3]
        hist_error = self.root_states_stored[:, 2, :5] - (self.commands[:,2:3] - 0.08)
        passed_ids = torch.all(hist_error>0, dim=-1)
        bounding_box_cons = rewed_ids * passed_ids
        # hip dof_pos limitaion condition:
        hip_torque_termi = torch.logical_and(torch.any(self.dof_pos[:,:] < self.dof_pos_limits[:, 0], dim=-1), torch.any(self.dof_pos[:,:] > self.dof_pos_limits[:, 1], dim=-1))
        #
        collision_cutoff = torch.sum(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1), dim=-1) > 0.2
        roll_cutoff = torch.abs(self.roll) > 2.4
        #has_jumpd_cutoff = self.has_jumped
        #has_jumped_cutoff0 = torch.logical_and(self.has_jumped, self.max_height>0.61)
        #has_jumped_cutoff = torch.logical_and(has_jumped_cutoff0, self.root_states[:, 2]<0.39)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        # reset the one that is too far from the desired landing postion or the orientation is too far from desired yaw.
        #self.reset_buf[self.reset_idx_landing_error] = True         
        self.reset_buf |= self.time_out_buf
        #self.reset_buf |= hip_torque_termi
        self.reset_buf |= height_cutoff
        #self.reset_buf |= height_cutoff_up
        #self.reset_buf |= height_cutoff_low
        self.reset_buf |= roll_cutoff
        self.reset_buf |= collision_cutoff
        #self.reset_buf |= bounding_box_cons
        #self.reset_buf |= has_jumpd_cutoff
        #self.reset_buf |= has_jumped_cutoff  # for only jumped once purpose
        
        # 加上框的长度信息！！！

    def _reset_state_history(self,env_ids):
        """ Resets state history of selected environments"""
        hist_len = self.cfg.env.state_history_length
        self.base_lin_vel_history[env_ids] = 0.0#self.base_lin_vel[env_ids,:].repeat(1,hist_len) 
        self.base_ang_vel_history[env_ids] = 0.0#self.base_ang_vel[env_ids,:].repeat(1,hist_len)
        self.root_states_history[env_ids] = 0.#self.root_states[env_ids,0:3].repeat(1,hist_len)
        self.dof_pos_history[env_ids] = 0.0#self.default_dof_pos.repeat(1,hist_len)
        self.dof_vel_history[env_ids] = 0.0#self.dof_vel[env_ids,:].repeat(1,hist_len)
        self.actions_history[env_ids] = 0.
        self.contacts_history[env_ids] = 0.0#self.contacts[env_ids,:].repeat(1,hist_len)
        self.base_quat_history[env_ids] = torch.tensor([0.,0.,0.,1.],device=self.device).repeat(len(env_ids),hist_len)#self.root_states[env_ids,3:7].repeat(1,hist_len)
        self.ori_error_history[env_ids] = 0.

    def _reset_stored_states(self,env_ids):
        """ Resets stored states of selected environments"""
        # self.base_lin_vel_stored[env_ids] = self.base_lin_vel[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        # self.base_ang_vel_stored[env_ids] = self.base_ang_vel[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        self.root_states_stored[env_ids] = self.root_states[env_ids,:].repeat(1,1).unsqueeze(-1)
        # self.dof_pos_stored[env_ids] = self.dof_pos[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        # self.dof_vel_stored[env_ids] = 0.
        # self.actions_stored[env_ids] = 0.
        # self.contacts_stored[env_ids] = self.contacts[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        # self.base_quat_stored[env_ids] = self.base_quat[env_ids].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        # self.ori_error_stored[env_ids] = 0
        # # self.error_quat_stored[env_ids] = 0
        # self.force_sensor_stored[env_ids] = 0

        # self.pd_dof_pos_stored[env_ids] = self.dof_pos[env_ids,:].unsqueeze(-1)
        # self.pd_dof_vel_stored[env_ids] = self.dof_vel[env_ids,:].unsqueeze(-1)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._refresh_actor_dof_props(env_ids)

        self._resample_commands(env_ids)
        #self.commands[env_ids,:],self.command_vels[env_ids,:] = self._recompute_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_actions_2[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.max_height[env_ids] = self.base_init_state[2]#0.
        self.task_max_height[env_ids] = self.base_init_state[2]

        self.was_in_flight[env_ids] = False
        self.mid_air[env_ids] = False
        self.mid_air2[env_ids] = False
        self.first_flight[env_ids] = False
        self.command_mask[env_ids] = False
        self.landing_ids[env_ids] = False
        self.has_jumped[env_ids] = False
        self.last_has_jumped[env_ids] = False
        self.settled_after_init[env_ids] = False
        self.landing_poses[env_ids,:] = float('nan')#1e4 + self.root_states[env_ids,:7].clone()
        self.reset_idx_landing_error[env_ids] = False


        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.randomize_dof_props(env_ids)
        self.env_delay_steps[env_ids] = torch.randint(0, self.cfg.control.max_delay_steps + 1, (len(env_ids),), device=self.device)
        self.action_buffer[:, env_ids] = 0
        # self.has_jumped[env_ids] = False
        # self._has_jumped_rand_envs[env_ids] = False
        # RMA related reset buffer
        # self.proprio_hist_buf[env_ids] = 0

    def _store_states(self, env_ids):
        '''
        Store the current states in the state storage buffers
        '''
        
        # self.base_lin_vel_stored = torch.roll(self.base_lin_vel_stored, 1, dims=2)
        # self.base_lin_vel_stored[:,:,0] = self.base_lin_vel_history.clone() 

        # self.base_ang_vel_stored = torch.roll(self.base_ang_vel_stored, 1, dims=2)
        # self.base_ang_vel_stored[:,:,0] = self.base_ang_vel_history.clone()

        # only store when has_jumped triger:
        #print(self.root_states_stored.size())
        #print('--------env_ids is:', env_ids) # env_ids is for 4 feet respectively
        self.root_states_stored[env_ids,:,:] = torch.roll(self.root_states_stored[env_ids,:,:] , 1, dims=-1) # root_states_stored.size(): self.num_envs, 13, 10
        self.root_states_stored[env_ids,:,0] = self.root_states[env_ids,:]#self.root_states_history.clone()  # 0 是current state, 1—9 是previous state

        # self.dof_pos_stored = torch.roll(self.dof_pos_stored, 1, dims=2)
        # self.dof_pos_stored[:,:,0] = self.dof_pos_history.clone() 

        # self.dof_vel_stored = torch.roll(self.dof_vel_stored, 1, dims=2)
        # self.dof_vel_stored[:,:,0] = self.dof_vel_history.clone() 

        # self.actions_stored = torch.roll(self.actions_stored, 1, dims=2)
        # self.actions_stored[:,:,0] = self.actions_history.clone()

        # self.contacts_stored = torch.roll(self.contacts_stored, 1, dims=2) # self.contact_stored is (num_envs, 20*4feet, 1)
        # self.contacts_stored[:,:,0] = self.contacts_history.clone()

        # self.base_quat_stored = torch.roll(self.base_quat_stored, 1, dims=2)
        # self.base_quat_stored[:,:,0] = self.base_quat_history.clone() 

        # self.ori_error_stored = torch.roll(self.ori_error_stored, 1, dims=2)
        # self.ori_error_stored[:,:,0] = self.ori_error_history.clone()

        # # self.error_quat_stored = torch.roll(self.error_quat_stored, 1, dims=2)
        # # self.error_quat_stored[:,:,0] = self.error_quat_history.clone()
        
        # # self.force_sensor_stored = torch.roll(self.force_sensor_stored, 1, dims=3)
        # # self.force_sensor_stored[:,:,:,0] = self.force_sensor_readings_history.clone()
        
        # self.has_jumped_stored = torch.roll(self.has_jumped_stored, 1, dims=2)
        # self.has_jumped_stored[:,:,0] = self.has_jumped_history.clone()
    # def check_landing(self):
    #     landing = torch.all(self.last_contacts, dim=1)
    #     last_flight_ids = landing == False #num_envs, 4
    #     landed = torch.all(self.contacts, dim=1)
    #     now_land_ids = landed == True
    #     ids = torch.logical_and(last_flight_ids, now_land_ids) 
    #     return ids

    def check_landing(self):
        landing_ids = self.root_states_stored[:,2,1] > self.root_states_stored[:,2,0]
        ids = landing_ids * self.mid_air
        return ids

    def check_HJ_diff(self):
        env_ids = self.last_has_jumped == False
        return env_ids

    def check_has_jump_reset(self):
        contact_ids = torch.any(self.contact_filt,dim=1)
        contact_ids2 = torch.all(self.contact_filt,dim=1)
        self.task_max_height[contact_ids*self.has_jumped] = self.base_init_state[2]
        #print("-------------self.task_max_height is--------------", self.max_height)
        self.was_in_flight[contact_ids*self.has_jumped] = False
        self.landing_ids[contact_ids2] = False
        self.has_jumped[contact_ids*self.has_jumped] = False #這裏要加一個 last_contact 和現在contact都是 all true, 才轉爲False的判斷
        

    def compute_max_height(self):
        max_heights = torch.max(self.max_height[:],self.root_states[:, 2])

        #max_heights[self.was_in_flight] = 0
        max_heights[self.has_jumped] = 0
        return max_heights 

    def _compute_state_history(self):
        '''
        Compute the state history for the current timestep. This is done by shifting the current 
        history by one step and adding the current (possibly delayed) state to the first position.
        '''

        add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
                    
        base_lin_vel = self.base_lin_vel.clone() + \
         add_noise * noise_scales.lin_vel * noise_level * (2 * torch.rand_like(self.base_lin_vel) - 1) 
        
        base_ang_vel = self.base_ang_vel.clone() + \
            add_noise * noise_scales.ang_vel * noise_level * (2 * torch.rand_like(self.base_ang_vel) - 1)
        
        root_states = self.root_states.clone()

        dof_pos = self.dof_pos.clone() + \
            add_noise * noise_scales.dof_pos * noise_level * (2 * torch.rand_like(self.dof_pos) - 1)
        
        dof_vel = self.dof_vel.clone() + \
            add_noise * noise_scales.dof_vel * noise_level * (2 * torch.rand_like(self.dof_vel) - 1)
        
        actions = self.actions.clone()

        noise_prob = self.cfg.noise.noise_scales.contacts_noise_prob
        noise_prob_distr = torch.distributions.bernoulli.Bernoulli(torch.tensor([noise_prob],device=self.device))
        # If contact = 1 then 10% of the time it will be 0.
        # If contact = 0 then nothing happens
        contacts = self.contacts.clone() * \
            (1 - noise_prob_distr.sample((self.num_envs,4)).reshape(self.num_envs,-1))

        base_quat = self.base_quat.clone() + \
            add_noise * noise_scales.quat * noise_level * (2 * torch.rand_like(self.base_quat) - 1)
        
        ori_error = self.ori_error.clone().unsqueeze(-1) + \
            add_noise * noise_scales.ori_error * noise_level * (2 * torch.rand_like(self.ori_error.unsqueeze(-1)) - 1)
        
        # error_quat = self.error_quat.clone() + \
        #     add_noise * noise_scales.error_quat * noise_level * (2 * torch.rand_like(self.error_quat) - 1)
        
        has_jumped = self.has_jumped.clone().unsqueeze(-1)

        self.base_lin_vel_history = torch.roll(self.base_lin_vel_history, self.base_lin_vel.shape[-1], dims=1)
        self.base_lin_vel_history[:,0:self.base_lin_vel.shape[-1]] = base_lin_vel 
        
        self.base_ang_vel_history = torch.roll(self.base_ang_vel_history, self.base_ang_vel.shape[-1], dims=1)
        self.base_ang_vel_history[:,0:self.base_ang_vel.shape[-1]] = base_ang_vel
        
        self.root_states_history = torch.roll(self.root_states_history, 3, dims=1)
        self.root_states_history[:,0:3] = root_states[:,:3]
        
        self.dof_pos_history = torch.roll(self.dof_pos_history, self.dof_pos.shape[-1], dims=1)
        self.dof_pos_history[:,0:self.dof_pos.shape[-1]] = dof_pos 
        
        self.dof_vel_history = torch.roll(self.dof_vel_history, self.dof_vel.shape[-1], dims=1)
        self.dof_vel_history[:,0:self.dof_vel.shape[-1]] = dof_vel 

        self.actions_history = torch.roll(self.actions_history, self.actions.shape[-1], dims=1)
        self.actions_history[:,0:self.actions.shape[-1]] = actions
        
        self.contacts_history = torch.roll(self.contacts_history, self.contacts.shape[-1], dims=1) # size: num_envs, 20*4; type=bool
        self.contacts_history[:, 0:self.contacts.shape[-1]] = contacts # 把self.contact的内容传进self.contact_history
        
        self.base_quat_history = torch.roll(self.base_quat_history, self.base_quat.shape[-1], dims=1)
        self.base_quat_history[:,0:4] = base_quat 
        
        self.ori_error_history = torch.roll(self.ori_error_history, self.ori_error.shape[-1], dims=1)
        self.ori_error_history[:,0] = ori_error.squeeze(-1)
        
        # self.error_quat_history = torch.roll(self.error_quat_history, self.error_quat.shape[-1], dims=1)
        # self.error_quat_history[:,0:4] = error_quat 

        self.has_jumped_history = torch.roll(self.has_jumped_history, self.has_jumped.shape[-1], dims=1)
        self.has_jumped_history[:,0] = has_jumped.squeeze(-1)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_eval(self):
        """Compute evals
        Reports eval function values, not used in training.
        """
        for i in range(len(self.eval_functions)):
            name = self.eval_names[i]
            rew = self.eval_functions[i]()
            self.eval_sums[name] += rew

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_vel = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        if self.cfg.env.train_type == "standard":
            noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.gravity * noise_level
            noise_vec[9:12] = 0.  # commands
            noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[36:48] = 0.  # previous actions
        else:
            noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.gravity * noise_level
            noise_vec[6:9] = 0.  # commands
            noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[33:45] = 0.  # previous actions
            # noise_vec[6:19] = 0.  # commands
            # noise_vec[19:31] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # noise_vec[31:43] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            # noise_vec[43:55] = 0.  # previous actions

        noise_vel[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vel[3:7] = 0
        noise_vel[7:11] = 0
        if self.enable_priv_measured_height:
            noise_vel[11:198] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        if self.cfg.env.measure_obs_heights:
            noise_vec[48:235] = (noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements)
        return noise_vec, noise_vel
    def compute_observations(self):
        """ Computes observations
        """

        if self.cfg.env.train_type == "standard":
            self.obs_buf = torch.cat((#self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.projected_gravity,
                                      self.commands[:, :3] * self.commands_scale,
                                      self.commands[:,3:4] * 2.0,
                                      #self.commands[:, :3] * self.commands_scale,
                                      self._get_dof_pos_obs(),
                                      self.dof_vel * self.obs_scales.dof_vel,
                                      self.actions,
                                      ), dim=-1)

        else:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.projected_gravity,# 代表了quad, imu输出的很准，不需要再预估了。
                                      #self.commands[:, :7],  # distance tracking
                                      #self.command_vels[:, :6], # distance tracking
                                      #self.commands[:, :4] * self.commands_scale, # add z_height tracking
                                      self.commands[:, :3] * self.commands_scale, # vel tracking
                                      self.commands[:,3:4] * 2.0, # desired COM heights command, 2.0 is to similar to the lin_vel scale 
                                      self._get_dof_pos_obs(),
                                      self.dof_vel * self.obs_scales.dof_vel,
                                      self.actions,
                                      #self.base_quat, # self.rootstates[:,3:7] add 4 dims
                                      ), dim=-1)  # normally do not change it. Because it's the 45-dim obs of actor.

# self.privileged_obs_buf means 私有变量。It's the obs of critics.
        self.privileged_obs_buf = self.root_states[:, 2:3]
        if self.enable_priv_Zheights_weights: #false. No weight prediction
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                self.priv_info_buf[:, 0:4], # for adding the weights, dim=4
                                                 ), dim=-1)
        if self.enable_priv_ZXYheights: # true, add the real ZXY-position values to the crtic.
            #self.privileged_obs_buf = self.root_states[:, 2:3]
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                self.root_states[:, 0:2], # for adding the weights, dim=4
                                                 ), dim=-1)  
        if self.enable_priv_feet_height:
             self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                  self.FL_feet_pos[:,2:3],   # choose 1 from dim=3 for each
                                                  self.FR_feet_pos[:,2:3],
                                                  self.RL_feet_pos[:,2:3],
                                                  self.RR_feet_pos[:,2:3],
                                                 ), dim=-1)      
        if self.enable_priv_ang_vel:
            #self.privileged_obs_buf = self.root_states[:, 10:13]
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                               self.root_states[:, 10:13], # for adding the weights, dim=4
                                                ), dim=-1)


        if self.enable_priv_enableMeasuredVel:
            # the following line is only for the vel_estimator
            vel = self.base_lin_vel * self.obs_scales.lin_vel # dim=3  
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.
            #self.privileged_obs_buf = vel
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 vel,   #dim =3     # only for the vel_estimator
                                                 self.foot_height, # dim=4
                                                 contact,          # dim=4
                                                 ), dim=-1)

        if self.enable_priv_measured_height:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.34 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 heights, #187
                                                 ), dim=-1)
            if self.cfg.env.measure_obs_heights:
                self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1) # obs_buf is for the actor input

        if self.enable_priv_disturbance_force:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.disturbance_force, # dim = 2
                                                 ), dim=-1)


        # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        #     self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_noise_vel


        if self.cfg.env.train_type in ("Dream", "RMA"):#or "GenHis":
            # deal with normal observation, do sliding window
            prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone() # 4096, 5, 45+1

            # concatenate to get full history
            cur_obs_buf = self.obs_buf.clone().unsqueeze(1)

            self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

            # refill the initialized buffers
            # if reset, then the history buffer are all filled with the current observation
            at_reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.obs_buf_lag_history[at_reset_env_ids, :, :] = self.obs_buf[at_reset_env_ids].unsqueeze(1)

            self.proprio_hist_buf = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone() 

    def _get_dof_pos_obs(self):
        """Return scaled joint-position observations.

        Tasks with continuous joints may override this without changing the
        observation layout used by the shared environment implementation.
        """
        return (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos



    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh", "QRC","obs_stone","stone", "stair"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type == "QRC":
            self._create_trimesh()    
        elif mesh_type == "obs_stone":
            self._create_trimesh()      
        elif mesh_type == "stone":
            self._create_trimesh()
        elif mesh_type == "stair":
            self._create_trimesh()         
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_body_props(self, props, env_id):
        # randomize Base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
            if self.cfg.env.train_type in ("RMA", "GenHis"):
                self._update_priv_buf(env_id=env_id, name='base_mass', value=props[0].mass,
                                      lower=rng[0], upper=rng[1])
        # randomize limb mass:
        if self.cfg.env.train_type in ("RMA", "GenHis"):
            limb_mass = [props[1].mass, props[2].mass, props[3].mass]  # FL hip/thigh/calf
            self._update_priv_buf(env_id=env_id, name='limb_mass', value=limb_mass,
                                  lower=0.1, upper=4.0)
        return props

    def _refresh_actor_dof_props(self, env_ids):

        for env_id in env_ids:
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                dof_props["friction"][i] = self.joint_friction_coeffs[env_id, 0]

            self.gym.set_actor_dof_properties(self.envs[env_id], 0, dof_props)

    def _process_rigid_com_props(self, props, env_id):
        # randomize center
        if self.cfg.domain_rand.randomize_center:
            center = self.cfg.domain_rand.added_center_range
            com = [np.random.uniform(center[0], center[1]), np.random.uniform(center[0], center[1])]
            props[0].com.x, props[0].com.y = com
            if self.cfg.env.train_type == "RMA":
                self._update_priv_buf(env_id=env_id, name='center', value=com,
                                      lower=center[0], upper=center[1])
        return props


    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            friction_range = self.cfg.domain_rand.friction_range
            if env_id == 0:
                # prepare friction randomization
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            if self.cfg.env.train_type == "RMA":
                self._update_priv_buf(env_id=env_id, name='friction', value=props[0].friction,
                                      lower=friction_range[0], upper=friction_range[1])

        return props
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF
        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id
        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item() 
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item() #* 0.8
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            # print("dof_pos_limits lower:", self.dof_pos_limits[:, 0])
            # print("dof_pos_limits up:", self.dof_pos_limits[:, 1])
        if self.cfg.domain_rand.randomize_motor_strength:
            motor_strength = []
            for i in range(self.num_dofs):
                name = self.dof_names[i]
                found = False
                rand_motor_strength = np.random.uniform(self.cfg.domain_rand.added_motor_strength[0],
                                                        self.cfg.domain_rand.added_motor_strength[1])
                for dof_name in self.cfg.control.stiffness.keys():

                    if dof_name in name:
                        if self.cfg.control.control_type == "POSE":
                            props['driveMode'][i] = gymapi.DOF_MODE_POS
                            props['stiffness'][i] = self.cfg.control.stiffness[dof_name] * rand_motor_strength  # self.Kp
                            props['damping'][i] = self.cfg.control.damping[dof_name] * rand_motor_strength  # self.Kd
                        elif self.cfg.control.control_type == "actuator_net":

                            rand_motor_offset = np.random.uniform(self.cfg.domain_rand.added_Motor_OffsetRange[0],
                                              self.cfg.domain_rand.added_Motor_OffsetRange[1])

                            self.motor_offsets[env_id][i] = rand_motor_offset
                            self.motor_strengths[env_id][i] = rand_motor_strength

                        else:
                            self.p_gains[env_id][i] = self.cfg.control.stiffness[dof_name] * rand_motor_strength
                            self.d_gains[env_id][i] = self.cfg.control.damping[dof_name] * rand_motor_strength
                        found = True
                if not found:
                    self.p_gains[i] = 0.0
                    self.d_gains[i] = 0.0
                    if self.cfg.control.control_type in ["P", "V"]:
                        print(f"PD gain of joint {name} were not defined, setting them to zero")
                motor_strength.append(rand_motor_strength)

            if self.cfg.env.train_type == "RMA":
                self._update_priv_buf(env_id=env_id, name='motor_strength', value=motor_strength,
                                      lower=self.cfg.domain_rand.added_motor_strength[0],
                                      upper=self.cfg.domain_rand.added_motor_strength[1])


        return props

    def randomize_dof_props(self, env_ids):
               
        if self.cfg.domain_rand.randomize_joint_friction:
            joint_friction_range = self.cfg.domain_rand.ranges.joint_friction_range
            self.joint_friction_coeffs[env_ids] = torch_rand_float(joint_friction_range[0], joint_friction_range[1], (len(env_ids), 1), device=self.device)


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(env_ids)
        #self.commands[env_ids,:], self.command_vels[env_ids,:] = self._recompute_commands(env_ids)
        #if self.cfg.commands.heading_command:
        forward = quat_apply(self.base_quat, self.forward_vec) #
        self.forward = forward
            #heading = torch.atan2(forward[:, 1], forward[:, 0])
            #self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self.disturbance_force = self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        choice = torch.randint(0,2, (len(env_ids), 1),device=self.device).squeeze(1)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1) #* choice
        #self.commands[choice==0, 0] = 0
        y_choice = torch.where(choice>0,0,1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1) #* y_choice
        #self.commands[choice==1, 1] = 0 
            #old:
        if not self.cfg.commands.bool_jump: # default to be false
            if self.cfg.commands.height_command:        
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["height_z"][0],
                                                        self.command_ranges["height_z"][1],
                                                        (len(env_ids), 1),
                                                        device=self.device).squeeze(1)
                temp = self.commands[env_ids, 3] 
                trot_ids = temp < 0.46
                temp[trot_ids] = 0.32
                lower0 = 0.46 <= temp
                upper0 = temp < 0.6           
                temp[lower0 * upper0] = 0.5#0.55#0.55 #0.50
            
                lower1 = 0.6 <= temp
                upper1 = temp < 0.85
                temp[lower1 * upper1] = 0.68#0.72#0.75 #0.66 #0.68 # 0.7
            
                self.commands[env_ids, 3] = temp
        # new；
        else:
            z_jump = (torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1) < self.cfg.commands.jump_prob).float()
            self.commands_z [env_ids, 0] = torch_rand_float(self.command_ranges["z_jump"][0], self.command_ranges["z_jump"][1], (len(env_ids), 1), device=self.device).squeeze(1) * z_jump
            self.commands_z [env_ids, 0] += torch_rand_float(self.command_ranges["z_normal"][0], self.command_ranges["z_normal"][1], (len(env_ids), 1), device=self.device).squeeze(1) * (1 - z_jump)
            temp = self.commands_z[env_ids, 0]
            lower0 = 0.46 <= temp
            upper0 = temp < 0.6           
            temp[lower0 * upper0] = 0.56#0.50 #0.50

            lower1 = 0.6 <= temp
            upper1 = temp < 0.76
            temp[lower1 * upper1] = 0.8#0.68
            self.commands_z[env_ids, 0] = temp
            if self.cfg.commands.zero_v_cmd_normal:
                self.commands[env_ids, :3] *= z_jump.unsqueeze(1)
            self.commands[env_ids, 3] = self.commands_z[env_ids, 0] 

            #temp = self.commands[env_ids, 3]
            
            # temp[temp < 0.45] = 0.36
            # lower00 = 0.45 <= temp
            # upper00 = temp < 0.6
            # temp[lower00 * upper00] = 0.55

# typical height selection jump aliengo:
            # temp[temp<0.6] = 0.55  
            # #temp[temp<0.65] = 0.55 

            # lower0 = 0.6 <= temp
            # #lower0 = 0.65 <= temp
            # upper0 = temp < 0.75 
            # temp[lower0 * upper0] = 0.7#0.8 #

            # lower1 = 0.75 <= temp
            # upper1 = temp < 1.0
            # temp[lower1 * upper1] = 0.85#0.8 # 

# long jump policy:
            # temp[temp<0.6] = 0.55  

            # lower0 = 0.6 <= temp
            # upper0 = temp < 0.75
            # temp[lower0 * upper0] = 0.68

            # lower1 = 0.75 <= temp
            # upper1 = temp < 0.9
            # temp[lower1 * upper1] = 0.8       
# go2 jump:
            # height_sample = self.root_states[env_ids, 2]
            # temp[temp<=0.45] = height_sample[temp<=0.45]

            # lower0 = 0.46 <= temp
            # upper0 = temp < 0.6           
            # temp[lower0 * upper0] = 0.50 #0.50

            # lower1 = 0.6 <= temp
            # upper1 = temp < 0.76
            # temp[lower1 * upper1] = 0.68 #0.66 #0.68 # 0.7
 
            # self.commands[env_ids, 3] = temp
            

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        if self.cfg.commands.tracking_z: # bool variable to regulate the jumping process.
            self.commands[env_ids, 5] = 1.* torch.randint(self.command_ranges["vel_z_bool"][0],
                                                         self.command_ranges["vel_z_bool"][1]+1,
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
            
    def _recompute_commands(self,env_ids):
        """ Recompute relative distance for the jumps:

    #     """
        commands = torch.zeros_like(self.commands)
        command_vels = torch.zeros_like(self.command_vels)
    # dx dy dz is the variance on the command distance which helps to build the curriculum
        dx = torch.zeros((self.num_envs, 1), device=self.device).flatten()
        dy = torch.zeros((self.num_envs, 1), device=self.device).flatten()
        dz = torch.zeros((self.num_envs, 1), device=self.device).flatten()
        
        # up_jump_envs = self.up_jump_distribution.sample((len(env_ids),1)).flatten()
        # env_ids_up_jump = env_ids[up_jump_envs==1]
        # # For now only change the pos components:
        # dx[env_ids] = torch_rand_float(self.pos_command_variation[0,0], self.pos_command_variation[1,0], (len(env_ids), 1), device=self.device).flatten()
        # dy[env_ids] = torch_rand_float(self.pos_command_variation[0,1], self.pos_command_variation[1,1], (len(env_ids), 1), device=self.device).flatten()
        dx[env_ids] = torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).flatten()
        dy[env_ids] = torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).flatten()        
        # # dz = torch_rand_float(self.pos_command_variation[0,2], self.pos_command_variation[1,2], (len(env_ids), 1), device=self.device)
        # dx[env_ids_up_jump] = 0.0
        # dy[env_ids_up_jump] = 0.0
        dz[env_ids] += self.env_origins[env_ids,2]  #其实用不到

    # define the command 0~2 here: which are the. The commanded distance range for robots to track:
        commands[env_ids, 0] = dx[env_ids].squeeze() + self.command_distances["x"] # dx is randomized during each episode, which provides the disturbance for desired jumping distance.
        commands[env_ids, 1] = dy[env_ids].squeeze() + self.command_distances["y"] # cfg.commands.ranges + cfg.commands.distances
        commands[env_ids, 2] = dz[env_ids].squeeze() + self.command_distances["z"]


        des_angles_euler = torch.zeros((self.num_envs,3),device=self.device)
        commands[env_ids, 7::] = 0.

        # Convert to quaternion:
        # des_angles_euler = torch.tensor(self.command_distances["des_angles_euler"]).view(3,1)
        # Desired yaw depends on the heading between starting point and goal
        initial_yaw = wrap_to_pi(get_euler_xyz(self.root_states[:,3:7])[2])
        
        des_angles_euler[:,2] = wrap_to_pi(torch.atan2(commands[:,1],commands[:,0]) - initial_yaw)
        if self.cfg.commands.randomize_yaw:
            des_angles_euler[:,2] += torch_rand_float(-np.pi/2, np.pi/2, (self.num_envs, 1), device=self.device).flatten()
            des_angles_euler[:,2] = wrap_to_pi(des_angles_euler[:,2])
            # des_angles_euler[:,2] = torch.clip(des_angles_euler[:,2], -np.pi/2, np.pi/2)
        if self.cfg.commands.distances.des_yaw is not None:
            des_angles_euler[:,2] = self.cfg.commands.distances.des_yaw

        self.des_angles_euler[env_ids] = des_angles_euler[env_ids]
        desired_quat = quat_from_euler_xyz(des_angles_euler[:,0],des_angles_euler[:,1],des_angles_euler[:,2])#.squeeze()
        # if self.cfg.commands.distances.des_yaw==0:
        #     commands[env_ids, 3:7] = 0
        # else:
        commands[env_ids, 3] =  desired_quat[env_ids,0]
        commands[env_ids, 4] =  desired_quat[env_ids,1]
        commands[env_ids, 5] =  desired_quat[env_ids,2]
        commands[env_ids, 6] =  desired_quat[env_ids,3]

        # Update the desired velocities:

        # These have been derived based on best fit line on joint friction vs flight time from the upwards jump:
        a = -4.4207
        b = 0.5563
        flight_time = self.joint_friction_coeffs[env_ids] * a + b
        command_vels[env_ids,0:3] = commands[env_ids,0:3]/(flight_time) # 速度=路程/时间
        command_vels[env_ids,3:6] = des_angles_euler[env_ids,:]/(flight_time)

        return commands[env_ids],command_vels[env_ids]
 

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        # print('sfs',self.p_gains, self.d_gains)
        if control_type == "P":
            if self.cfg.domain_rand.added_lag_timesteps:
                self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
                self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
            else:
                self.joint_pos_target = actions_scaled + self.default_dof_pos
            torques = (
                     self.p_gains * (self.joint_pos_target - self.dof_pos)
                     - self.d_gains * self.dof_vel)
            # torques = (
            #         self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
            #         - self.d_gains * self.dof_vel
            
        elif control_type == "POSE":
            torques = (
                    self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                    - self.d_gains * self.dof_vel
            )            

        elif control_type == "actuator_net1":

            if self.cfg.domain_rand.added_lag_timesteps:
                self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
                self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
            else:
                self.joint_pos_target = actions_scaled + self.default_dof_pos

            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel

            # torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,self.joint_pos_err_last_4,
            #                                 self.joint_vel, self.joint_vel_last, self.joint_vel_last_last, self.joint_vel_last_4)

            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            
            self.joint_pos_err_last_4 = torch.clone(self.joint_pos_err_last_4)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last_last = torch.clone(self.joint_vel_last_last)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
            # scale the output
            torques = torques * self.motor_strengths

        elif control_type == "actuator_net2":

            if self.cfg.domain_rand.added_lag_timesteps:
                self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
                self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
            else:
                self.joint_pos_target = actions_scaled + self.default_dof_pos

            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets # 4096, 12
            self.joint_vel = self.dof_vel  # 4096, 12

            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last, self.joint_pos_err_last_4, self.joint_pos_err_last_5,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last, self.joint_vel_last_4, self.joint_vel_last_5) # add the joint_enc here
            self.joint_pos_err_last_5 = torch.clone(self.joint_pos_err_last_4)
            self.joint_pos_err_last_4 = torch.clone(self.joint_pos_err_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_5 = torch.clone(self.joint_vel_last_4)
            self.joint_vel_last_4 = torch.clone(self.joint_vel_last_last)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
            # scale the output
            torques = torques * self.motor_strengths

        elif control_type == "V":
            torques = (
                    self.p_gains * (actions_scaled - self.dof_vel)
                    - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits) # 

    def _update_morph_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.morphology_const_info_dict[name]
        if type(value) is list:
            value = to_torch(value, dtype=torch.float, device=self.device)
        if type(lower) is list or type(upper) is list:
            lower = to_torch(lower, dtype=torch.float, device=self.device)
            upper = to_torch(upper, dtype=torch.float, device=self.device)
        if lower is not None and upper is not None:
            value = (value - lower) / (upper - lower)
        self.morph_priv_info_buf[env_id, s:e] = value

    def _compute_poses(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale  # wo - current pos is better than w - current pos
        target_poses = actions_scaled + self.default_dof_pos
        return torch.clip(target_poses, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:cfg.control.1.5 x default positions.
        Velocities are set to zero.
        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets Base position based on the curriculum
            Selects randomized Base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        env_ids_all = env_ids.clone()
        # self.initial_root_states[env_ids_all,:] = self.root_states[env_ids_all,:].clone()
        # Base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state #[0, 0, 0.42]
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -0.8, 0.8, (len(env_ids), 2),device=self.device)
            # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # Base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        
        self.initial_root_states[env_ids_all,:] = self.root_states[env_ids_all,:].clone()        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32),
                                                     len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized Base velocity.
        """
        #env_ids = self.has_jumped.clone()
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                                                    device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        return self.root_states[:, 7:9] # dim=2

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # if self.cfg.rewards.scales.tracking_lin_vel>0 and torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
        #         self.reward_scales["tracking_lin_vel"]:
        #     self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
        #                                                   -self.cfg.commands.max_curriculum, 0.)
        #     self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
        #                                                   self.cfg.commands.max_curriculum)
        # for sepcifically difficult task, set 15% as the threshold.
        if self.cfg.rewards.scales.tracking_lin_vel>0 and torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.5 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = self.command_ranges["lin_vel_x"][0] + 1.0
            self.command_ranges["lin_vel_x"][1] = self.command_ranges["lin_vel_x"][1] + 1.0       

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        _fsdata= self.gym.acquire_dof_force_tensor(self.sim)


        if self.cfg.control.control_type == "POSE":
            # Note: Position control
            torques = self.gym.acquire_dof_force_tensor(self.sim)
            self.target_poses = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
            self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_actions)

            # self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            # self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        else:
            self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                       requires_grad=False)

        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.fsdata = gymtorch.wrap_tensor(_fsdata)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0] # 4096,12
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1] # 4096,12
        self.base_quat = self.root_states[:, 3:7]
        #print('fsdata value is:', self.fsdata)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis
        #self.contacts = torch.ones(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        # initialize some data used later on
        self.common_step_counter = 0
        self.commands_z  = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        self.extras = {}
        self.noise_scale_vec,self.noise_noise_vel = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # Note: torque get directly from gym

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_actions_2 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)


        #### add last_information ##################
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_forward = torch.zeros_like(quat_apply(self.base_quat, self.forward_vec))


        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.command_vels = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, z vel
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.des_angles_euler = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.task_max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.mid_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.mid_air2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.command_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        #up_jump_prob = self.cfg.commands.upward_jump_probability
        #self.up_jump_distribution = torch.distributions.bernoulli.Bernoulli(torch.tensor([up_jump_prob],device=self.device))
        self.landing_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.settled_after_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.settled_after_init_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self._has_jumped_switched_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        
        #self.contacts_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*len(self.feet_indices),self.cfg.env.state_stored_length, dtype=torch.bool, device=self.device, requires_grad=False)
        #self.contacts_stored[:,:,:] = self.contacts.repeat(1,self.cfg.env.state_history_length).view(self.num_envs, self.cfg.env.state_history_length*len(self.feet_indices),1)

        #self.contacts_history = torch.zeros(self.num_envs, len(self.feet_indices)*self.cfg.env.state_history_length, dtype=torch.bool, device=self.device, requires_grad=False)

        self.root_states_stored = torch.zeros(self.num_envs, 13, 10, dtype=torch.float, device=self.device, requires_grad=False) # 13 is the length of root_states
        self.root_states_stored[:, :, 0] = self.root_states# state_history is default to be 20 in curriculum_based jumping 
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) # 10:13 is the ang_vel in the world_frame  # roll, pitch, yaw 
        self.feet_pos = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec) # num_envs, 3
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.initial_root_states = torch.zeros_like(self.root_states)
        self.landing_poses = torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_idx_landing_error = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.disturbance_force = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)  # x vel, y vel

        self.all_feet_up = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.body_max_heights = -torch.ones(self.num_envs, device=self.device)

        foot_height  = self.rigid_body_state[:, self.feet_indices, 2:3]
        foot_vel = self.rigid_body_state[:, self.feet_indices, 7:9]

        self.foot_height = torch.cat((foot_height[:, 0, ],
                                      foot_height[:, 1, ],
                                      foot_height[:, 2, ],
                                      foot_height[:, 3, ],
                                      ), dim=-1)
        self.foot_vel = torch.cat((foot_vel[:, 0, ],
                                   foot_vel[:, 1, ],
                                   foot_vel[:, 2, ],
                                   foot_vel[:, 3, ],
                                      ), dim=-1)
        # actuator delay
        # Initialize delay steps for each environment
        self.env_delay_steps = torch.randint(0, self.cfg.control.max_delay_steps + 1, (self.num_envs,), device=self.device)
        # Initialize action buffer
        max_delay = self.cfg.control.max_delay_steps
        self.action_buffer = torch.zeros((max_delay + 1, self.num_envs, self.num_actions), device=self.device)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_peak = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) 

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            #angle = self.cfg.init_state.default_joint_angles_peak[name]
            #angle_peak = self.cfg.init_state.default_joint_angles_peak[name]
            self.default_dof_pos[i] = angle
            #self.default_dof_pos_peak[i] = angle_peak


        # if self.cfg.control.control_type == "POSE":
        #     for i in range(self.num_dofs):
        #         name = self.dof_names[i]
        #         angle = self.cfg.init_state.default_joint_angles[name]
        #         self.default_dof_pos[i] = angle
        # else:
        #     for i in range(self.num_dofs):
        #         name = self.dof_names[i]
        #
        #         angle = self.cfg.init_state.default_joint_angles[name]
        #         self.default_dof_pos[i] = angle
        #         found = False
        #         for dof_name in self.cfg.control.stiffness.keys():
        #             if dof_name in name:
        #                 self.p_gains[i] = self.cfg.control.stiffness[dof_name]
        #                 self.d_gains[i] = self.cfg.control.damping[dof_name]
        #                 found = True
        #         if not found:
        #             self.p_gains[i] = 0.
        #             self.d_gains[i] = 0.
        #             if self.cfg.control.control_type in ["P", "V"]:
        #                 print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_peak = self.default_dof_pos_peak.unsqueeze(0)


        # Additionally initialize actuator network hidden state tensors
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.added_lag_timesteps + 1)]

        # if self.cfg.control.control_type == "POSE":
        # init pos err and vel buffer(12 DOF)
        self.joint_pos_err_last_5 = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_pos_err_last_4 = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_vel_last_5 = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_vel_last_4 = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.joint_friction_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))


        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
    def _prepare_eval_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales
        to_remove = []
        for name, scale in self.eval_scales.items():
            if not scale:
                to_remove.append(name)
        for name in to_remove:
            self.eval_scales.pop(name)

        # prepare list of functions
        self.eval_functions = []
        self.eval_names = []
        for name, scale in self.eval_scales.items():
            #print(name, scale)
            if name == "termination":
                continue
            self.eval_names.append(name)
            name = "_eval_" + name
            self.eval_functions.append(getattr(self, name))

        # reward episode sums
        self.eval_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.eval_scales.keys()
        }

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        #self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        #print("----------------------------")
        #print("rigid body names:", body_names)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        #print("------------------------------")
        #print("feet names are:", feet_names)
        self.foot_name = feet_names
        self.body_names = body_names


        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.coms = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                          requires_grad=False)#[]
        self.ests = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                          requires_grad=False)#[]        
        # if self.cfg.control.control_type != "actuator_network":
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False) # 4096,12

        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                               requires_grad=False)
        self._init_custom_buffers__()        
        self.randomize_dof_props(torch.arange(self.num_envs, device=self.device))

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            #pos[0] += ((1 - (-1)) * torch.rand(1, device=self.device) + (-1))[0]#.squeeze(1)
            #pos[1] += ((1 - (-1)) * torch.rand(1, device=self.device) + (-1))[0]#.squeeze(1)
            #pos[2] = self.cfg.init_state.pos[2]
            start_pose.p = gymapi.Vec3(*pos)
            #print(start_pose.p)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            #self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)  # Note: important to read torque !!!!
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            body_props = self._process_rigid_body_props(body_props, i)
            self.mass = body_props[0].mass
            self.G = -9.81 * self.mass
            recompute_inertia = getattr(
                self.cfg.domain_rand, "recompute_inertia", True
            )
            self.gym.set_actor_rigid_body_properties(
                env_handle,
                actor_handle,
                body_props,
                recomputeInertia=recompute_inertia,
            )


            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self._refresh_actor_dof_props(torch.arange(self.num_envs, device=self.device))        
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)


        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i]) #['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh", "trimesh_on_stair", "stair"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            #self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float) # 10*40*8*2            
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level #5
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        #self.command_distances = class_to_dict(self.cfg.commands.distances)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s # totally 20s per episode
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt) # 20/0.02 = 1000
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self, estimations):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        np.set_printoptions(precision=4)

        # draw height lines
        if False: #self.cfg.terrain.measure_heights:
            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 2, 2, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                               self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
        com_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0.85, 0.1, 0.1))
        est_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0, 1, 0))
        com_proj_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0, 1, 0))
        contact_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0, 0, 1))
        cop_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0.85, 0.5, 0.1))
        for i in range(self.num_envs):
            # draw COM(= Base pose) and its projection
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            est_height = estimations[i][0].cpu().numpy()
            com = self.coms[i].cpu().numpy() + base_pos
            #est = est_height#self.ests[i].cpu().numpy() + est_height
            com_pose = gymapi.Transform(gymapi.Vec3(com[0], com[1], com[2]), r=None)
            com_proj_pose = gymapi.Transform(gymapi.Vec3(com[0], com[1], 0), r=None)
            est_pose = gymapi.Transform(gymapi.Vec3(com[0]+np.array(0.08), com[1]+np.array(0.08), est_height), r=None)
            #gymutil.draw_lines(com_geom, self.gym, self.viewer, self.envs[i], com_pose)
            #gymutil.draw_lines(est_geom, self.gym, self.viewer, self.envs[i], est_pose)
            #gymutil.draw_lines(com_proj_geom, self.gym, self.viewer, self.envs[i], com_proj_pose)
            
            # draw contact point and COP projection
            eef_state = self.rigid_body_state[i, self.feet_indices, :3]
            contact_idxs = (self.contact_forces[i, self.feet_indices, 2] > 1.).nonzero(as_tuple=False).flatten()
            contact_state = eef_state[contact_idxs]
            contact_force = self.contact_forces[i, self.feet_indices[contact_idxs], 2]

            for i_feet in range(contact_state.shape[0]):
                contact_pose = contact_state[i_feet, :]
                sphere_pose = gymapi.Transform(gymapi.Vec3(contact_pose[0], contact_pose[1], 0), r=None)
                gymutil.draw_lines(contact_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            # calculate and draw COP
            cop_sum = torch.sum(contact_state * contact_force.view(contact_force.shape[0], 1), dim=0)
            cop = cop_sum / torch.sum(contact_force)
            sphere_pose = gymapi.Transform(gymapi.Vec3(cop[0], cop[1], 0), r=None)
            #gymutil.draw_lines(cop_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            # # draw inverted pendulum
            # cop = cop.cpu().numpy()
            # self.gym.add_lines(self.viewer, self.envs[i], 1, [cop[0], cop[1], 0, com[0], com[1], com[2]], [0.85, 0.1, 0.5])

            # draw connected line between contact point (fault case and normal case)
            #self._draw_contact_polygon(contact_state, self.envs[i])
            self._draw_height_pillar(self.envs[i])
            self._draw_est_pillar(self.envs[i], estimations)

    def _draw_contact_polygon(self, contact_state, env_handle):
        contact_num = contact_state.shape[0]
        if contact_num >= 2:
            if contact_num == 4:  # switch the order of rectangle
                contact_state = contact_state[[0, 1, 3, 2], :]
            polygon_start = contact_state[0].cpu().numpy()

            width, n_lines = 0.01, 10  # make it thicker： width 越大， n_lines也要越多
            polygon_starts = []
            for i_line in range(n_lines):
                polygon_starts.append(polygon_start.copy())
                polygon_start += np.array([0, 0, width / n_lines])
            for i_feet in range(contact_num):
                polygon_end = contact_state[(i_feet + 1) % contact_num, :].cpu().numpy()

                polygon_ends = []
                polygon_vecs = []
                for i_line in range(n_lines):
                    polygon_ends.append(polygon_end.copy())
                    polygon_end += np.array([0, 0, width / n_lines])
                    polygon_vecs.append(
                        [polygon_starts[i_line][0], polygon_starts[i_line][1], polygon_starts[i_line][2],
                         polygon_ends[i_line][0], polygon_ends[i_line][1], polygon_ends[i_line][2]])
                self.gym.add_lines(self.viewer, env_handle, n_lines,
                                   polygon_vecs,
                                   n_lines * [0.85, 0.1, 0.1])

                polygon_starts = polygon_ends

    def _draw_height_pillar(self, env_handle):

        com_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0.85, 0.1, 0.1))

        for i in range(self.num_envs):
            # draw COM(= Base pose) and its projection
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            com = self.coms[i].cpu().numpy() + base_pos
            com_pose = gymapi.Transform(gymapi.Vec3(com[0], com[1]+0.12, com[2]), r=None)
            #gymutil.draw_lines(com_geom, self.gym, self.viewer, self.envs[i], com_pose)

            pillar_start = np.array([com[0],com[1]+0.12, 0])
                             
            width, n_lines = 0.03, 30  # make it thicker
            pillar_starts = []
            pillar_ends = []
            pillar_vecs = []
            for i_line in range(np.int(n_lines)):
                pillar_starts.append(pillar_start.copy())
                pillar_start += np.array([0 - width / n_lines, 0, 0])
            # for i_line in range(np.int(n_lines/2)):
            #     pillar_starts.append(pillar_start.copy())
            #     pillar_start += np.array([0 + width / n_lines, 0, 0])            
            
                pillar_end = pillar_start + np.array([0,0, com[2]])

            #for i_line in range(n_lines):
                pillar_ends.append(pillar_end.copy())
                pillar_end += np.array([0 - width / n_lines, 0, 0])
                pillar_vecs.append(
                        [pillar_starts[i_line][0], pillar_starts[i_line][1], pillar_starts[i_line][2],
                         pillar_ends[i_line][0], pillar_ends[i_line][1], pillar_ends[i_line][2]])
            self.gym.add_lines(self.viewer, env_handle, n_lines,
                                   pillar_vecs,
                                   n_lines * [0.85, 0.1, 0.1])

            #pillar_starts = pillar_ends
    def _draw_est_pillar(self, env_handle, estimations):

        est_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0, 1, 0))
        com_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0.85, 0.1, 0.1))

        for i in range(self.num_envs):
            # draw COM(= Base pose) and its projection
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            est_height = estimations[i][0].cpu().numpy()
            com = self.coms[i].cpu().numpy() + base_pos
            com_pose = gymapi.Transform(gymapi.Vec3(com[0], com[1], com[2]), r=None)
            est_pose = gymapi.Transform(gymapi.Vec3(com[0]+np.array(0.06), com[1]+np.array(0.12), est_height), r=None)
            #gymutil.draw_lines(est_geom, self.gym, self.viewer, self.envs[i], est_pose)

            pillar_start = np.array([com[0] + 0.06, com[1] + 0.12, 0])
                             
            width, n_lines = 0.03, 30  # make it thicker
            pillar_starts = []
            pillar_ends = []
            pillar_vecs = []
            for i_line in range(np.int(n_lines)):
                pillar_starts.append(pillar_start.copy())
                pillar_start += np.array([0 - width / n_lines, 0, 0])
            # for i_line in range(np.int(n_lines/2)):
            #     pillar_starts.append(pillar_start.copy())
            #     pillar_start += np.array([0 + width / n_lines, 0, 0])            
            
                pillar_end = pillar_start + np.array([0,0, est_height])

            #for i_line in range(n_lines):
                pillar_ends.append(pillar_end.copy())
                pillar_end += np.array([0 - width / n_lines, 0, 0])
                pillar_vecs.append(
                        [pillar_starts[i_line][0], pillar_starts[i_line][1], pillar_starts[i_line][2],
                         pillar_ends[i_line][0], pillar_ends[i_line][1], pillar_ends[i_line][2]])
            self.gym.add_lines(self.viewer, env_handle, n_lines,
                                   pillar_vecs,
                                   n_lines * [0, 1, 0])

            #pillar_starts = pillar_ends

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in Base frame)
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False) #4096*187*3
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the Base's position and rotated by the Base's yaw
        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.
        Raises:
            NameError: [description]
        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
            self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py] # rows * cols
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)        

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale # 4096*187

    def check_jump(self):
        """ Check if the robot has jumped
        """

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # contact_filt = contact
        contact_filt = torch.logical_or(contact, self.last_contacts) # Contact is true only if either current or previous contact was true
        self.contact_filt = contact_filt.clone() # Store it for the rewards that use it

        # Handle starting in mid-air (initialise in air):
        settled_after_init = torch.logical_and(torch.all(contact_filt,dim=1), self.root_states[:,2]<=0.34) # torch.all() outputs the dimension with all True.
        jump_filter = torch.all(~contact_filt, dim=1)#torch.logical_and(torch.all(~contact_filt, dim=1),self.root_states[:,2]>0.32) # If no contact for all 4 feet, jump is true
        jump_filter2 = torch.all(~contact, dim=1)
        # jump_filter: last 和current沒有一隻足碰到了地
        # torch.all() output True only if all are true. torch.any() output flase only if all are false.
        # jump_filter = torch.sum(contact_filt, dim=1) <= 2 # If more than  foot is in the air, jump has started

        self.mid_air = jump_filter.clone()
        self.mid_air2 = jump_filter2.clone() 

        idx_record_pose = torch.logical_and(settled_after_init,~self.settled_after_init)
        self.settled_after_init_timer[idx_record_pose] = self.episode_length_buf[idx_record_pose].clone()
        self.settled_after_init[settled_after_init] = True

        # Only consider in flight if robot has settled after initialisation and is in the air:
        # (only switched to true once for each robot per episode)
        self.was_in_flight[torch.logical_and(jump_filter,self.settled_after_init)] = True # If no contact for all 4 feet, robot is in flight

        # The robot has already jumped IFF it was previously in flight and has now landed:
        has_jumped = torch.logical_and(torch.any(contact_filt, dim=1), self.was_in_flight) 

        # Only count the first time flight is achieved:
        self.landing_poses[torch.logical_and(has_jumped,~self.has_jumped)] = self.root_states[torch.logical_and(has_jumped,~self.has_jumped),:7] #之前没跳，这个step刚跳起来的那些 
        #self.landing_foot_poses[torch.logical_and(has_jumped,~self.has_jumped)] = self.feet_pos[torch.logical_and(has_jumped,~self.has_jumped),:,:]

        # Only count the first time flight is achieved:        
        self.has_jumped[has_jumped] = True # record the idx which are has_jumped

        # 当前 episode 中第一次起跳后的腾空阶段。原代码直接读取
        # self.first_flight，但从未创建该张量，导致环境首次 reset 时崩溃。
        self.first_flight = torch.logical_and(self.mid_air, ~self.has_jumped)

        landing_ids = self.check_landing()
        self.landing_ids = landing_ids.clone()

        self.last_contacts = contact
        
        # for single jump
        self.command_mask = self.first_flight.clone()

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # lin_vel_error = torch.sum(torch.square(self.commands[self.mid_air, :2] - self.base_lin_vel[self.mid_air, :2]), dim=1)
        # for jumping:
        #lin_vel_error = torch.sum(torch.square(self.commands[self.mid_air, :2] - self.root_states[self.mid_air, 7:9]), dim=1)
        # for walking:
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim=1)
        #lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=-1)
    
        # for jumping:
        #rew[self.mid_air] = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        # for walking:
        rew = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return rew

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # for jumping:
        #ang_vel_error = torch.square(self.commands[self.mid_air, 2] - self.base_ang_vel[self.mid_air, 2])
        #for walking: 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        #command_ids = self.commands[:,3] < 0.45
        #rew[self.mid_air] = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma) 
        rew = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma) 
        #rew[command_ids] = 0
        return rew

    def _reward_jump_distance(self): # only compute for the ids whose dim1 and dim2 are different.
        error = torch.norm(self.root_states_stored[:,0:3,0] - self.root_states_stored[:,0:3,1], dim=1) 
        rew = torch.exp(error * 0.001)
        return rew

    def _reward_tracking_pitch_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 1])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)           

    # def _reward_lin_vel_z(self):
    #     # Penalize z axis Base linear velocity
    #     # return torch.square(self.base_lin_vel[:, 2])
    #     #return (self.commands[:, 4])*self.base_lin_vel[:, 2]
    #     return (self.commands[:, 4])*self.base_lin_vel[:, 2]

    def _reward_lin_vel_z_world(self):
        # z_vel_error = torch.square(self.commands[:, 5] - self.root_states[:, 9])
        # return torch.exp(-z_vel_error / self.cfg.rewards.tracking_sigma)
        return self.root_states[:, 9]
        #return (self.commands[:, 4])*self.root_states[:, 9]
    
    @no_jump
    def _reward_lin_vel_z(self):
        # return super()._reward_lin_vel_z()
        return torch.abs(self.base_lin_vel[:, 2])

    def _reward_lin_disz_world(self):
        env_ids = torch.logical_and(self.episode_length_buf == self.max_episode_length,self.has_jumped)
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        t = 0.5#self.max_episode_length_s/(2.5) #* self.root_states[:, 9] #self.commands[:, 5]
        g = -9.81 #m/s^2
        ids = ~self.has_jumped
        #print("no jumped agents:", torch.sum(ids))
        #predefined max_height is 0.7
        h = self.root_states[ids, 9] * t + 0.5 * g * (t**2)
        #print("h is:", h)
        error = h - 0.7
        rew[ids] = torch.exp(-torch.square(error))
        #print("max disz reward value:", torch.max(rew))
        return rew

    def _reward_ang_vel_xy(self):
        # Penalize xy axes Base angular velocity: roll and pitch
        #return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        # Penalize z axes Base angular velocity: yaw.
        return torch.square(self.base_ang_vel[:, 2])
    
    def _reward_tracking_yaw(self):
    #   rew = torch.exp(-torch.abs(self.commands[:, 2] - self.yaw))
    #   return rew
        env_ids = self.episode_length_buf == self.max_episode_length

        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            pass
        else:
            # Only give a reward for robots that have landed and are at the end of the episode:
            idx = env_ids * self.has_jumped
            #print("dim of cmd2 is:", self.commands[:, 2].size())
            #print("dim of yaw is:", self.yaw.size())
            rew[idx] = torch.exp(-torch.abs(self.commands[idx, 2] - self.yaw[idx]))

        return rew
    
    def _reward_tracking_pitch(self):

        env_ids = self.episode_length_buf == self.max_episode_length
    
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            pass
        else:
            # Only give a reward for robots that have landed and are at the end of the episode:
            idx = env_ids #* self.has_jumped
            #print("dim of cmd2 is:", self.commands[:, 2].size())
            #print("dim of yaw is:", self.yaw.size())
            rew[idx] = torch.exp(-torch.abs(self.commands[idx, 2] - self.pitch[idx]))

        return rew

    def _reward_task_pos(self): # jumping distance
        # Reward for completing the task
        
        env_ids = self.episode_length_buf == self.max_episode_length
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        # Base position relative to initial states:
        rel_root_states = self.landing_poses[:,:2] - self.initial_root_states[:,:2]

        tracking_error = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        tracking_error = torch.linalg.norm(rel_root_states[:] - self.commands[:, :2],dim=1) # push the robot to approach the desired landing position
        # Check which envs have actually jumped (and not just been initialised at an already "jumped" state)
        has_jumped_idx = self.has_jumped

        max_tracking_error = 0.5 #(self.cfg.env.reset_landing_error * (self.commands[:,:2])).clip(min=0.1)

        self.reset_idx_landing_error[torch.logical_and(has_jumped_idx, tracking_error>max_tracking_error)] = True
        
        idx = has_jumped_idx
        #rew[idx] = torch.exp(-torch.square(tracking_error[idx])/0.05)
        rew[idx] = torch.exp(-tracking_error[idx]/0.05)
        #print('tracking position error mean is :', torch.mean(tracking_error[idx]*0.001))
        #print('task_pos reward error mean is :', torch.mean(rew))
        return rew

    def _reward_task_ori(self): # jumping gesture
        # Reward for completing the task
        #env_ids = self.episode_length_buf == self.max_episode_length # 尽可能只跳一次

        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        # quat_landing = self.landing_poses[:, 3:7]
        # quat_des = self.commands[:, 3:7]
        # ori_tracking_error = quat_distance(quat_landing, quat_des)

        _,_,yaw_landing = get_euler_xyz(self.landing_poses[:, 3:7])
        _,_,yaw_des = get_euler_xyz(self.commands[:, 3:7])

        ori_tracking_error_yaw = torch.abs(wrap_to_pi(yaw_landing-yaw_des)) # minimize the error towards desired value.

        # Check which envs have actually jumped (and not just been initialised at an already "jumped" state)
        has_jumped_idx = self.has_jumped
        #self.reset_idx_landing_error[torch.logical_and(has_jumped_idx,ori_tracking_error_yaw>0.5)] = True

        # if torch.all(env_ids == False): # if no env is done return 0 reward for all
        #     pass
        # else:
        #     # Only give a reward for robots that have landed and are at the end of the episode:
        #     idx = env_ids * has_jumped_idx
            
        #     rew[idx] = torch.exp(-torch.square(ori_tracking_error_yaw[idx])/0.05)
        idx = has_jumped_idx
        #rew[idx] = torch.exp(-torch.square(ori_tracking_error_yaw[idx])/0.05)
        rew[idx] = torch.exp(-ori_tracking_error_yaw[idx]/0.05)

        return rew 

    def _reward_vel_switch(self):
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        trackz_agent_mask = lin_vel_error > 0.04 # select which agents' x vel tracking errors are too large, reward its vel_z
        lin_velz_error = torch.square(self.base_lin_vel[:, 2] - self.commands[:, 0])
        trackz = trackz_agent_mask * torch.exp(-lin_velz_error/self.cfg.rewards.tracking_sigma)
        trackx = ~trackz_agent_mask * torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        rew = trackz+trackx
        return rew

    def _reward_reaction_force(self):
        upward_agent = self.root_states[:, 9] > 0
        down_agent = self.root_states[:, 9] < 0
         
        return 

    def _reward_height_track(self): # root_states to track. maybe useful
        # Reward for max height achieved during the episode:
        # env_ids = torch.logical_and(self.episode_length_buf == self.max_episode_length,self.has_jumped) # reward mask.
        # #env_ids = self.episode_length_buf == self.max_episode_length
        # rew  = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        # if torch.all(env_ids == False): # if no env is done return 0 reward for all
        #     return rew
    

        # max_height_reward = (self.max_height[env_ids] - self.cfg.rewards.base_height_target) # encourage the max_height to approach 0.9

        # rew = torch.exp(max_height_reward)
        # mid_air condition:
        # rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # h_error = torch.square(self.root_states[self.mid_air,2] - self.commands[self.mid_air, 3])
        # rew[self.mid_air] = torch.exp(-h_error / self.cfg.rewards.tracking_sigma)
        # normal condition:
        h_error = torch.square(self.root_states[:,2] - self.commands[:, 3])
        rew = torch.exp(-h_error / self.cfg.rewards.tracking_sigma)
        #rew[self.commands[:,3] < 0.45] = 0  ###
        
        return rew



    def _reward_max_track(self):
        rew  = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        max_height_reward = (self.max_height[self.mid_air] - self.commands[self.mid_air, 3]) # encourage the max_height to approach 0.9 # aliengo 0.85
# only the one that has_jumped and finish the whole episode can track the max_height reward.
        rew[self.mid_air] = torch.exp(-torch.square(max_height_reward)/0.05)

        return rew #* (self.commands[:, 4])

    def _reward_task_max_height(self): # maybe useful
        # Reward for max height achieved during the episode:
# for no lower bond:
        # env_ids = torch.logical_and(self.episode_length_buf == self.max_episode_length,self.has_jumped) # compute the ids that has jumped during the whole epoch.
# for hard lower bond:
        # env_ids = torch.logical_or(self.episode_length_buf == self.max_episode_length,
        #           torch.logical_and(self.reset_buf, self.episode_length_buf < self.max_episode_length))
        
        # rew  = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        # if torch.all(env_ids == False): # if no env is done return 0 reward for all: if all elements in env_ids all False, the torch.all() will be True. As long as one element is True, the torch.all() value will be False.
        #     return rew
#         max_height_reward = (self.task_max_height[env_ids] - self.commands[env_ids, 3]) # encourage the max_height to approach 0.9 # aliengo 0.85
# # only the one that has_jumped and finish the whole episode can track the max_height reward.
#         rew[env_ids] = torch.exp(-torch.square(max_height_reward)/0.05)

        max_height_reward = (self.task_max_height - self.commands[:, 3]) # encourage the max_height to approach 0.9 # aliengo 0.85
# only the one that has_jumped and finish the whole episode can track the max_height reward.
        rew = torch.exp(-torch.square(max_height_reward)/0.05)
        walking_ids = self.commands[:, 3] < 0.38
        rew[walking_ids] = 0
    
        return rew #* (self.commands[:, 4]) 
    
    def _reward_base_height_stance(self):
        # Reward feet height
        base_height = self.root_states[:, 2]
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        # Get the height of the terrain at the base position: (to offset the global base height):
        #heights = self.get_terrain_height(self.root_states[:,:2]).flatten()
        heights = 0
        base_height_stance = (base_height - heights - 0.38)[self.has_jumped] # aliengo 0.39, go1 0.31

        squat_idx = torch.logical_and(~self.mid_air,~self.has_jumped)
        base_height_squat = (self.root_states[squat_idx, 2] - 0.27)

        rew[squat_idx] = 0.6*torch.exp(-torch.square(base_height_squat)/0.001)
        rew[self.has_jumped] =  torch.exp(-torch.square(base_height_stance)/0.005)
        
        return rew 

    def _reward_jumping(self):
        # Reward if the robot has jumped in the episode:
        # env_ids = torch.logical_or(self.episode_length_buf == self.max_episode_length,
        #           torch.logical_and(self.reset_buf, self.episode_length_buf < self.max_episode_length))

        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        
        #rew[env_ids * self.has_jumped * self.max_height>0.4] = 10        
        #rew[self.has_jumped * self.max_height>self.commands[:, 3]] = 1 #100  # aliengo 0.75
        up_bond = self.commands[:, 3]+0.04
        low_bond = self.commands[:, 3]-0.04
        rewed_ids = torch.logical_and(torch.logical_and(self.has_jumped, self.task_max_height>low_bond), self.task_max_height<up_bond)
        #rewed_ids = torch.logical_and(torch.logical_and(self.has_jumped, self.max_height>low_bond), self.max_height<up_bond)
        #rew[rewed_ids * env_ids] = 1
        rew[rewed_ids] = 1
        walking_ids = self.commands[:, 3] < 0.38
        rew[walking_ids] = 0
        #rew[self.commands[:, 3]<0.45] = 0  ### only when stick on ground
        return rew

        
    def _reward_constrained_jumping(self):
        # 加上框的长度信息 bounding box
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        up_bond = self.commands[:, 3]+0.03 #6
        low_bond = self.commands[:, 3]-0.04 #5
        # bounding box term:
        rewed_ids = torch.logical_and(self.task_max_height>low_bond, self.task_max_height<up_bond)
        hist_error = self.root_states_stored[:, 2, :5] - (self.commands[:,2:3] - 0.08)
        passed_ids = torch.all(hist_error>0, dim=-1)
        bounding_box_cons = rewed_ids * passed_ids
        rew[rewed_ids] = 1
        #rew[bounding_box_cons] = 10        
        return rew                             
    
    def _reward_base_height(self):
        # Penalize Base height away from target
        # print('sdfswwf',self.root_states[:, 2].unsqueeze(1).shape, self.measured_heights.shape, (self.root_states[:, 2].unsqueeze(1) - self.measured_heights).shape)

        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        #print('measured heights sizes are:', self.measured_heights.size()) # num_envs * 187
        #base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
        # base_rew = base_height - 0.32
        # return torch.exp(-torch.square(base_rew))

    def _reward_tracking_pos(self):
        #self.measured_heights.size()) # num_envs * 187
        envs = torch.range(0,self.num_envs-1,dtype=int)
        tracked_z = torch.max(self.measured_heights, dim=-1).values #4096 # as the to be tracked z value
        index = torch.argmax(self.measured_heights, dim=-1) # 4096 * 1(0~186)
        # x_index = index//17 #4096
        # y_index = index%17  #4096
        points = self._init_height_points()
        x_temp = points[:,:,0]
        tracked_x = x_temp[envs,index] #4096
        y_temp = points[:,:,1]
        tracked_y = y_temp[envs,index] #4096
        # tracking point should in the front of the base
        in_the_front = tracked_x >= self.root_states[:,0] # delete the points behind the base
        tracking_error = torch.square(0.4+tracked_z-self.root_states[:,2])#torch.square(tracked_x-self.root_states[:,0])+torch.square(tracked_y-self.root_states[:,1])+torch.square(0.37+tracked_z-self.root_states[:,2])
        rew = torch.exp(-tracking_error)*in_the_front
        return rew
    
    def _reward_default_pose(self):
        # Penalise large actions:

        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        angle_diff = torch.square(self.dof_pos - self.default_dof_pos) # self.default_dof_pos is 0, 0.72, 1.4144
        #angle_diff[:, 0::3] *= 10 
        #angle_diff[self.mid_air + self.has_jumped,0::3] *= 10 # For forward hips
        #angle_diff[self.mid_air + self.has_jumped, 0] *= 10 
        #angle_diff[self.mid_air + self.has_jumped, 9] *= 10 
        #angle_diff[self.mid_air + self.has_jumped, 6] *= 10
        # emphasis on the hip pose:
        #angle_diff[:, 9] *= 5
        # angle_diff[self.mid_air2, 1::4] = 0
        # angle_diff[self.mid_air2, 7::10] = 0
        # angle_diff[self.mid_air2, 2::5] = 0
        # angle_diff[self.mid_air2, 8::11] = 0
        #angle_diff[self.mid_air2, 6] *= 5
        #angle_diff[self.mid_air2, 9] *= 5
        #angle_diff[:, 6] *= 5

        rew = torch.exp(torch.sum(angle_diff,dim=1)*0.01)
        #rew[self.landing_ids] *= 3.0 # emphasis on the has_jumped one.
        #rew[self.root_states[:,2]>0.34] = 0

        #when training the walking policy comment the below part: 
        #rew[self.mid_air2] = 0
        return rew # 6 and 12

    def _reward_tracking_feet_pos(self):
        envs = torch.range(0,self.num_envs-1,dtype=int)
        tracked_z = torch.max(self.measured_heights, dim=-1).values #4096 # as the to be tracked z value
        index = torch.argmax(self.measured_heights, dim=-1) # 4096 * 1(0~186)
        points = self._init_height_points()
        x_temp = points[:,:,0]
        tracked_x = x_temp[envs,index] #4096
        y_temp = points[:,:,1]
        tracked_y = y_temp[envs,index] #4096
        # tracking point should in the front of the base
        in_the_front = tracked_x > self.root_states[:,0] # delete the points behind the base
        in_the_air1 = self.root_states[:,0] > 0.5
        contact = torch.sum(self.contact_forces[:, self.feet_indices, 2],dim=1)
        contact_off = contact < 1
        in_the_air = in_the_air1 * contact_off
        #FL
        FLdesired_feet_pos_z = tracked_z+0.37 - 0.154 # the desired feet position in the world frame.
        FLdesired_feet_pos_x = tracked_x + 0.209
        FLdesired_feet_pos_y = tracked_y + 0.150
        error1 = torch.square(FLdesired_feet_pos_z-self.feet_pos[:, 0, 2])#torch.square(FLdesired_feet_pos_x-self.feet_pos[:, 0, 0])+torch.square(FLdesired_feet_pos_y-self.feet_pos[:, 0, 1])+torch.square(FLdesired_feet_pos_z-self.feet_pos[:, 0, 2])
        rew1 = torch.exp(-error1)
        #print('error1 mean ', torch.mean(error1))
        #print("rew1 mean", torch.mean(rew1))
        #FR
        FRdesired_feet_pos_z = tracked_z+0.37 - 0.154
        FRdesired_feet_pos_x = tracked_x + 0.209
        FRdesired_feet_pos_y = tracked_y - 0.150
        error2 = torch.square(FRdesired_feet_pos_z-self.feet_pos[:, 1, 2])#torch.square(FRdesired_feet_pos_x-self.feet_pos[:, 0, 0])+torch.square(FRdesired_feet_pos_y-self.feet_pos[:, 0, 1])+torch.square(FRdesired_feet_pos_z-self.feet_pos[:, 0, 2])
        rew2 = torch.exp(-error2)
        #RL
        RLdesired_feet_pos_z = tracked_z+0.37 - 0.154
        RLdesired_feet_pos_x = tracked_x - 0.272
        RLdesired_feet_pos_y = tracked_y + 0.150
        error3 = torch.square(RLdesired_feet_pos_z-self.feet_pos[:, 2, 2])#torch.square(RLdesired_feet_pos_x-self.feet_pos[:, 0, 0])+torch.square(RLdesired_feet_pos_y-self.feet_pos[:, 0, 1])+torch.square(RLdesired_feet_pos_z-self.feet_pos[:, 0, 2])
        rew3 = torch.exp(-error3)
        #RR
        RRdesired_feet_pos_z = tracked_z+0.37 - 0.154
        RRdesired_feet_pos_x = tracked_x - 0.272
        RRdesired_feet_pos_y = tracked_y - 0.150
        error4 = torch.square(RRdesired_feet_pos_z-self.feet_pos[:, 3, 2])#torch.square(RRdesired_feet_pos_x-self.feet_pos[:, 0, 0])+torch.square(RRdesired_feet_pos_y-self.feet_pos[:, 0, 1])+torch.square(RRdesired_feet_pos_z-self.feet_pos[:, 0, 2])
        rew4 = torch.exp(-error4)
        rew = rew1+rew2+rew3+rew4
        #rew = rew*in_the_front
        #print('rew mean', torch.mean(rew*in_the_air))
        return rew*in_the_air

    def _reward_orientation(self):
        # Penalize non flat Base orientation
        #print("orientation reward value is: ", torch.exp(self.projected_gravity[:, 0]))
        #return torch.exp(self.projected_gravity[:, 0])
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_FL_phase_height(self):
        feet_height = self.feet_pos[:, 0, 2]
        max_feet = feet_height - 0.70
        return torch.exp(-torch.square(max_feet)/0.05)
        #return torch.max(feet_height)

    def _reward_FR_phase_height(self):
        feet_height = self.feet_pos[:, 1, 2]
        max_feet = feet_height - 0.70
        return torch.exp(-torch.square(max_feet)/0.05)
        #return torch.max(feet_height)
    
    def _reward_RL_phase_height(self):
        feet_height = self.feet_pos[:, 2, 2]
        max_feet = feet_height - 0.70
        return torch.exp(-torch.square(max_feet)/0.05)        
        #return torch.max(feet_height)
    
    def _reward_RR_phase_height(self):
        feet_height = self.feet_pos[:, 3, 2]
        max_feet = feet_height - 0.70
        return torch.exp(-torch.square(max_feet)/0.05)
        #return torch.max(feet_height)
    
    def _reward_not_sync_contact(self):
        # keep contact with ground
        sync_contact_rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # check sync contact
        front_contacts = (self.contacts[:, 0] != self.contacts[:, 1])
        back_contacts = (self.contacts[:, 2] != self.contacts[:, 3])
        sync_contacts = front_contacts | back_contacts
        sync_contact_rew[sync_contacts] = 1.0
        return sync_contact_rew
    
    def _reward_sync_contact(self):
        # keep contact with ground
        sync_contact_rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # check sync contact
        flrr_contacts = (self.contacts[:, 0] == self.contacts[:, 3]) #2
        frrl_contacts = (self.contacts[:, 1] == self.contacts[:, 2]) #3
        trot_contacts = flrr_contacts & frrl_contacts
        sync_contact_rew[trot_contacts] = 1.0
        return sync_contact_rew    
    
    def _reward_trot_contact(self):
        sync_contact_rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # diagnol feet contact at the same schedule:
        flrr_contacts = (self.contacts[:, 0] == self.contacts[:, 3])
        frrl_contacts = (self.contacts[:, 1] == self.contacts[:, 2])
        trot_contacts = flrr_contacts & frrl_contacts
        # front/back feet don't contact at the same schedule:
        front_contacts = (self.contacts[:, 0] != self.contacts[:, 1])
        back_contacts = (self.contacts[:, 2] != self.contacts[:, 3])
        sync_contacts = front_contacts | back_contacts

        total_contact_ids = trot_contacts & sync_contacts
        sync_contact_rew[total_contact_ids] = 1.0
        jumping_ids = self.commands[:, 3] > 0.4
        sync_contact_rew [jumping_ids] = 0
        return sync_contact_rew  

    def _reward_flfr_gait_diff(self): # left joints difference
        return torch.sum(torch.square(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [3, 4, 5]]), dim=1)
    
    def _reward_flfr_gait_diff2(self): # left joints difference
        return torch.sum(torch.square(self.dof_vel[:, [0, 1, 2]] - self.dof_vel[:, [3, 4, 5]]), dim=1)

    def _reward_rlrr_gait_diff(self): # left joints difference
        return torch.sum(torch.square(self.dof_pos[:, [6, 7, 8]] - self.dof_pos[:, [9, 10, 11]]), dim=1)

    def _reward_rlrr_gait_diff2(self): # left joints difference
        return torch.sum(torch.square(self.dof_vel[:, [6, 7, 8]] - self.dof_vel[:, [9, 10, 11]]), dim=1)
    
    def _reward_flrl_gait_diff(self): # left joints difference
        return torch.sum(torch.square(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [6, 7, 8]]), dim=1)
    
    def _reward_frrr_gait_diff(self): # left joints difference
        return torch.sum(torch.square(self.dof_pos[:, [3, 4, 5]] - self.dof_pos[:, [9, 10, 11]]), dim=1)


    def _reward_flrl_gait_diff(self): # left joints difference
        return torch.sum(torch.square(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [6, 7, 8]]), dim=1)

    def _reward_rlrr_gait_diff(self): # left joints difference
        return torch.sum(torch.square(self.dof_pos[:, [3, 4, 5]] - self.dof_pos[:, [9, 10, 11]]), dim=1)

# for troting gait:
    def _reward_flrr_gait_diff(self): # left joints difference
        rew = torch.sum(torch.square(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [9, 10, 11]]), dim=1)
        walking_ids = self.commands[:, 3] < 0.38
        rew[~walking_ids] = 0
        return rew
    
    def _reward_flrr_gait_diff2(self): # left joints difference
        rew = torch.sum(torch.square(self.dof_vel[:, [0, 1, 2]] - self.dof_vel[:, [9, 10, 11]]), dim=1)
        walking_ids = self.commands[:, 3] < 0.38
        rew[~walking_ids] = 0
        return rew

    def _reward_frrl_gait_diff(self): # left joints difference
        rew = torch.sum(torch.square(self.dof_pos[:, [3, 4, 5]] - self.dof_pos[:, [6, 7, 8]]), dim=1)
        walking_ids = self.commands[:, 3] < 0.38
        rew[~walking_ids] = 0
        return rew

    def _reward_frrl_gait_diff2(self): # left joints difference
        rew = torch.sum(torch.square(self.dof_vel[:, [3, 4, 5]] - self.dof_vel[:, [6, 7, 8]]), dim=1)
        walking_ids = self.commands[:, 3] < 0.38
        rew[~walking_ids] = 0
        return rew

    def _reward_torques(self):
        # Penalize torques
        rew = torch.sum(torch.square(self.torques), dim=1)
        #thigh torques:
        #rew[:,[0,3,6,9]] *= 2
        #rew[:,[1,4,7,10]] *= 5
        return rew

    def _reward_hip_torques(self):
        Torques = torch.square(self.torques)
        thighH = Torques[:,[0,3,6,9]] * 2 
        rew = torch.sum(thighH, dim=1)
        return rew

    def _reward_thigh_torques(self):
        Torques = torch.square(self.torques)
        thighT = Torques[:,[1,4,7,10]] * 5
        rew = torch.sum(thighT, dim=1)
        return rew
    
    def _reward_calf_torques(self):
        Torques = torch.square(self.torques)
        thighC = Torques[:,[2,5,8,11]]
        rew = torch.sum(thighC, dim=1)
        return rew    

    def _reward_motion(self):
        # cosmetic penalty for motion
        return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)


    # ***************** energy disspation ***************
    def _reward_energy(self):
        # Penalize energy
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time_base(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact,
                                dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact
        return rew_airTime
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
                torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_distance(self):
        feet_relative = self.feet_pos[:, :, :3] - self.root_states[:, :3].unsqueeze(1)
        feet_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        for i in range(4):
            feet_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, feet_relative[:,i,:])
                 
        # feet_pos_ini = torch.tensor(self.cfg.init_state.rel_foot_pos).to(self.device).transpose(1,0).view(1,4,3)
        # feet_pos_des = feet_pos_ini.clone()
        # # In mid-air and above 0.45m height, track close to body (otheriwse track normal):
        # #feet_pos_des[:,:,2 ]= -0.17 # this is the deisred foot position, aliengo
        # feet_pos_des[:,:,2 ]= -0.13 # -0.15 for go1&2
            
        feet_pos_ini = torch.tensor(self.cfg.init_state.rel_foot_pos_peak).to(self.device).transpose(1,0).view(1,4,3)
        feet_pos_des = feet_pos_ini.clone()
        #feet_error = feet_body_frame[:,:,2] - feet_pos_des[:,:,2] # let foot pos in body frame closed to the desired foot position
        feet_error = torch.linalg.norm(feet_body_frame - feet_pos_des,dim=-1)
        rew = torch.exp(-torch.sum(torch.square(feet_error), dim=-1))

        #feet_error_landing = torch.linalg.norm(feet_body_frame - feet_pos_ini,dim=-1)
        #rew_landing = torch.exp(-torch.sum(torch.square(feet_error_landing), dim=-1))
        # Only reward if in mid_air, hasn't jumped and height is above 0.45
        # base_height = self.root_states[:,2]
        #rew[base_height<=0.36] = rew_landing[base_height<=0.36]
        #rew[base_height<=0.34] = 0.0
        rew[~self.mid_air] = 0.0 # only reward the agents in the mid-air
        #rew[self.has_jumped] = 0.0
        #print("foot clearance reward is: ", rew)
        return rew
    
    def _reward_feet_pos(self): # smooth the feet gesture in the mid_air
        #desired_feet_pos_z = 0.347 * self.root_states[:, 2] - 0.438 # the delta_z = f(Zcom) = a*Zcom+b (fixed 0.8m desired height)
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        #desired_feet_pos_z = 0.15/(self.commands[:,3]-0.31) * base_height + (0.31*self.commands[:,3]-0.0496)/(0.31-self.commands[:,3]) # aliengo
        desired_feet_pos_z = 0.183/((self.commands[:,3]-0.1)-0.315) * base_height + (0.315*(self.commands[:,3]-0.1)-0.0416)/(0.315-(self.commands[:,3]-0.1)) # go2
        feet_relative = self.feet_pos[:, :, :3] - self.root_states[:, :3].unsqueeze(1)
        feet_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        for i in range(4):
            feet_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, feet_relative[:,i,:]) #(4096,4,3)
 
        feet_pos_ini = torch.tensor(self.cfg.init_state.rel_foot_pos_peak).to(self.device).transpose(1,0).view(1,4,3)
        feet_pos_des = feet_pos_ini.clone() #(1,4,3)
        feet_pos_des = feet_pos_des.cpu().data.numpy()
        Go2_IK = RobotIK(Go2)  
        jp,jv = Go2_IK.computeIK(np.array([feet_pos_des[0,0,0], feet_pos_des[0,0,1], feet_pos_des[0,0,2], feet_pos_des[0,1,0], feet_pos_des[0,1,1], feet_pos_des[0,1,2], feet_pos_des[0,2,0], feet_pos_des[0,2,1], feet_pos_des[0,2,2], feet_pos_des[0,3,0], feet_pos_des[0,3,1], feet_pos_des[0,3,2]]),
                 np.array([0,0,0,0,0,0,0,0,0,0,0,0])) # 最后十二维改成着陆时，足端相对base的速度。
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        jp = torch.from_numpy(jp)
        jp = jp.to(self.device)
        angle_diff = torch.square(self.dof_pos - jp)
        angle_diff[:, 0] *= 10
        angle_diff[:, 3] *= 10
        angle_diff[:, 6] *= 10   
        angle_diff[:, 9] *= 10
        rew = torch.exp(-torch.sum(angle_diff, dim=1) * 0.01)

        rew[~self.mid_air] = 0.0 # only reward the agents in the mid-air        
        return rew
    
    def _reward_tracking_air_angle(self):
        # Consider how to implement IK here: mid_air 都track同一个angle
        Go2_IK = RobotIK(Go2)
        jp,jv = Go2_IK.computeIK(np.array([0.201, 0.144, -0.146, 0.201, -0.144, -0.146,-0.279, 0.144, -0.146,-0.279, -0.144, -0.146]),
                 np.array([0,0,0,0,0,0,0,0,0,0,0,0]))
        # jp,jv = Go2_IK.computeIK(np.array([0.199, 0.150, -0.179, 0.199, -0.150, -0.179,-0.188, 0.150, -0.179,-0.188, -0.150, -0.179]),
        #          np.array([0,0,0,0,0,0,0,0,0,0,0,0]))
        # jp,jv = Go2_IK.computeIK(np.array([0.232, 0.148, -0.112, 0.232, -0.148, -0.112,-0.155, 0.148, -0.115,-0.155, -0.148, -0.115]),
        #          np.array([0,0,0,0,0,0,0,0,0,0,0,0]))
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        jp = torch.from_numpy(jp)
        jp = jp.to(self.device)
        angle_diff = torch.square(self.dof_pos - jp)
        # emphasis on two rear thigh joints
        angle_diff[:, 6] *= 10
        #angle_diff[:, 9] *= 5#10   
        angle_diff[:, 9] *= 10
        # angle_diff[:, 6] *= 5#5#10              
        #angle_diff[:, 7] *= 1.#5#10
        #angle_diff[:, 10] *= 1.#5#10
        rew = torch.exp(torch.sum(angle_diff,dim=1)*0.01)
        #rew[~self.mid_air2] = 0.
        rew[~self.mid_air] = 0.
        #rew[self.root_states[:,2]<0.34] = 0
        return rew

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
                    torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_motion_base(self):
        # cosmetic penalty for motion
        return torch.sum(torch.square(self.dof_pos[:,: ] - self.default_dof_pos[:, :]), dim=1)


    def _reward_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)

    def _reward_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - self.default_dof_pos[:, [1, 4, 7, 10]]), dim=1)

    def _reward_calf_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [2, 5, 8, 11]] - self.default_dof_pos[:, [2, 5, 8, 11]]), dim=1)

    ############## Motion Functions ############
    def _reward_f_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [0, 3]] - self.default_dof_pos[:, [0, 3]]), dim=1)

    def _reward_r_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [6, 9]] - self.default_dof_pos[:, [6, 9]]), dim=1)

    def _reward_f_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [1, 4]] - self.default_dof_pos[:, [1, 4]]), dim=1)

    def _reward_r_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [7, 10]] - self.default_dof_pos[:, [7, 10]]), dim=1)

    def _reward_f_calf_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [2, 5]] - self.default_dof_pos[:, [2, 5]]), dim=1)

    def _reward_r_calf_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [8, 11]] - self.default_dof_pos[:, [8, 11]]), dim=1)

    def _reward_f_hip_motion_height(self):
        # cosmetic penalty for hip motion
        new_default_dof_pos = self.default_dof_pos.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        new_default_dof_pos_height=self.default_dof_pos_peak.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        default_dof_pos_height = ((1-self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)* new_default_dof_pos+ ((self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)*new_default_dof_pos_height  
        return torch.sum(torch.abs(self.dof_pos[:, [0, 3]] - default_dof_pos_height[:, [0, 3]]), dim=1)

    def _reward_r_hip_motion_height(self):
        # cosmetic penalty for hip motion
        new_default_dof_pos = self.default_dof_pos.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        new_default_dof_pos_height=self.default_dof_pos_peak.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        default_dof_pos_height = ((1-self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)* new_default_dof_pos+ ((self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)*new_default_dof_pos_height  
        return torch.sum(torch.abs(self.dof_pos[:, [6, 9]] - default_dof_pos_height[:, [6, 9]]), dim=1)

    def _reward_f_thigh_motion_height(self):
        # cosmetic penalty for hip motion
        new_default_dof_pos = self.default_dof_pos.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        new_default_dof_pos_height=self.default_dof_pos_peak.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        default_dof_pos_height = ((1-self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)* new_default_dof_pos+ ((self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)*new_default_dof_pos_height  
        return torch.sum(torch.abs(self.dof_pos[:, [1, 4]] - default_dof_pos_height[:, [1, 4]]), dim=1)

    def _reward_r_thigh_motion_height(self):
        # cosmetic penalty for hip motion
        new_default_dof_pos = self.default_dof_pos.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        new_default_dof_pos_height=self.default_dof_pos_peak.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        default_dof_pos_height = ((1-self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)* new_default_dof_pos+ ((self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)*new_default_dof_pos_height  
        return torch.sum(torch.abs(self.dof_pos[:, [7, 10]] - default_dof_pos_height[:, [7, 10]]), dim=1)

    def _reward_f_calf_motion_height(self):
        # cosmetic penalty for hip motion
        new_default_dof_pos = self.default_dof_pos.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        new_default_dof_pos_height=self.default_dof_pos_peak.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        default_dof_pos_height = ((1-self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)* new_default_dof_pos+ ((self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)*new_default_dof_pos_height  
        return torch.sum(torch.abs(self.dof_pos[:, [2, 5]] - default_dof_pos_height[:, [2, 5]]), dim=1)

    def _reward_r_calf_motion_height(self):
        # cosmetic penalty for hip motion
        #print("base height is:", self.root_states[:,2])
        new_default_dof_pos = self.default_dof_pos.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        new_default_dof_pos_height=self.default_dof_pos_peak.squeeze(0).expand(self.cfg.env.num_envs, self.num_dof)
        default_dof_pos_height = ((1-self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)* new_default_dof_pos+ ((self.root_states[:,2].unsqueeze(1)/self.cfg.rewards.base_height_target)**2)*new_default_dof_pos_height  
        return torch.sum(torch.abs(self.dof_pos[:, [8, 11]] - default_dof_pos_height[:, [8, 11]]), dim=1)

    # imitation reward: consider root_pos, feet_pos, joint_angle
    def _reward_imitation_root_pos(self):
        #ref_data[0:3] # the refernce xyz for root position
        ref_num = 53#ref_data.shape[0]
        ref = torch.tensor(ref_data, dtype=torch.float, device=self.device)
        #reference = torch.tensor(ref_data[self.episode_length_buf%ref_num, 0:3], dtype=torch.float, device=self.device).unsqueeze(0)
        #reference = torch.tensor(ref_data[self.episode_length_buf%ref_num, 0:3]).unsqueeze(0)
        reference = ref[self.episode_length_buf%ref_num, 0:3]
        root_pos_error = torch.sum(torch.square(self.root_states[:, 0:3] - reference), dim=1)
        rew = torch.exp(-root_pos_error)
        return rew
    
    def _reward_imitation_joint_angle(self):
        #ref_data[7:19] # the refernce xyz for root position
        ref_num = 53#ref_data.shape[0]
        ref = torch.tensor(ref_data, dtype=torch.float, device=self.device)
        reference = ref[self.episode_length_buf%ref_num, 7:19]
        #reference = torch.tensor(ref_data[self.episode_length_buf%ref_num, 7:19]).unsqueeze(0)
        joint_angle_error = torch.sum(torch.square(self.dof_pos - reference), dim=1)
        rew = torch.exp(joint_angle_error)
        return rew
    
    def _reward_imitation_feet_pos(self):
        #ref_data[] # the refernce xyz for feet position
        ref_num = 53#ref_data.shape[0]
        ref = torch.tensor(ref_data, dtype=torch.float, device=self.device)
        reference = ref[self.episode_length_buf%ref_num, 19:31]
        #print("reference shape is :", reference.size())
        #reference = torch.tensor(ref_data[self.episode_length_buf%ref_num, 19:31]).unsqueeze(0) # the feet coordinates (Fl FR RL RR, xyz) in body_frame
        feet_relative = self.feet_pos[:, :, :3] - self.root_states[:, :3].unsqueeze(1)
        feet_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        for i in range(4):
            feet_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, feet_relative[:,i,:])
        feet_pos_error_FL = torch.sum(torch.square(feet_body_frame[:,0,:] - reference[:,0:3]), dim=1)
        feet_pos_error_FR = torch.sum(torch.square(feet_body_frame[:,1,:] - reference[:,3:6]), dim=1)
        feet_pos_error_RL = torch.sum(torch.square(feet_body_frame[:,2,:] - reference[:,6:9]), dim=1)
        feet_pos_error_RR = torch.sum(torch.square(feet_body_frame[:,3,:] - reference[:,9:12]), dim=1)
        error = feet_pos_error_FL +feet_pos_error_FR + feet_pos_error_RL + feet_pos_error_RR
        rew = torch.exp(error)
        return rew

    def _reward_tracking_z_jump(self):
        # encourage jumping higher than command z
        need_to_jump = self.commands_z.squeeze(1) > 0.5
        global_height = self.root_states[:, 2]
        ground_heights = 0 #TODO if use terrain
        body_heights = global_height - ground_heights
        land_mask = torch.any(self.contact_filt, dim=1) # num_envs * num_feet -> num_envs
        first_contact = (self.all_feet_up) * land_mask
        self.all_feet_up[:] = True
        biggers = body_heights > self.body_max_heights
        self.body_max_heights[biggers] = body_heights[biggers]
        threshold = self.cfg.rewards.jump_height_threshold
        rew_z_error = torch.sum((self.body_max_heights-threshold).clip(max=self.commands_z-threshold) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_z_error = first_contact * torch.exp(-torch.square(self.body_max_heights - self.commands_z.squeeze(1))/self.cfg.rewards.tracking_sigma)
        rew_jump = first_contact * (self.body_max_heights > threshold) * self.cfg.rewards.jump_weight
        rew_leave = (~land_mask) * self.cfg.rewards.jump_weight * 0.05
        self.body_max_heights[land_mask] = -1.
        self.all_feet_up *= ~land_mask
        return (rew_jump + rew_leave + rew_z_error) * need_to_jump - (rew_jump + rew_leave) * (~need_to_jump)

    def _reward_feet_angle_limit(self):
        # Penalize feet angle
        feet_ori = self.rigid_body_state[:, self.feet_indices, 3:7].clone().reshape(-1, 4)
        body_ori = self.root_states[:, 3:7].repeat(1, self.feet_indices.shape[0]).clone().reshape(-1, 4)
        feet_ori = quaternion_multiply(quaternion_inverse(body_ori), feet_ori) # in body frame
        vec = torch.tensor([0., 0., 1.], device=self.device, requires_grad=False).repeat(feet_ori.shape[0], 1)
        vec = quaternion_apply(feet_ori, vec)
        vec = vec.view(-1, 4, 3)
        
        hind_limit_up = self.cfg.rewards.hind_feet_z_axis_x_limit_upper # z axis of hind feet's x component should be less than limit
        front_limit_up = self.cfg.rewards.front_feet_z_axis_x_limit_upper 
        hind_limit_down = self.cfg.rewards.hind_feet_z_axis_x_limit_lower
        front_limit_down = self.cfg.rewards.front_feet_z_axis_x_limit_lower
        limit_down = torch.tensor([front_limit_down, front_limit_down, hind_limit_down, hind_limit_down], device=self.device, requires_grad=False).repeat(self.num_envs, 1)
        limit_up = torch.tensor([front_limit_up, front_limit_up, hind_limit_up, hind_limit_up], device=self.device, requires_grad=False).repeat(self.num_envs, 1)
        x_component = vec[:, :, 0]
        rwd1 = torch.sum((x_component - limit_up).clip(min=0.), dim=1)
        rwd2 = torch.sum((limit_down - x_component).clip(min=0.), dim=1)        
        
        hind_limit_up = self.cfg.rewards.hind_feet_z_axis_y_limit_upper
        front_limit_up = self.cfg.rewards.front_feet_z_axis_y_limit_upper
        hind_limit_down = self.cfg.rewards.hind_feet_z_axis_y_limit_lower
        front_limit_down = self.cfg.rewards.front_feet_z_axis_y_limit_lower
        limit_down = torch.tensor([front_limit_down, front_limit_down, hind_limit_down, hind_limit_down], device=self.device, requires_grad=False).repeat(self.num_envs, 1)
        limit_up = torch.tensor([front_limit_up, front_limit_up, hind_limit_up, hind_limit_up], device=self.device, requires_grad=False).repeat(self.num_envs, 1)
        rwd3 = torch.sum((vec[:, :, 1] - limit_up).clip(min=0.), dim=1)
        rwd4 = torch.sum((limit_down - vec[:, :, 1]).clip(min=0.), dim=1)
        
        rwd = rwd1 + rwd2 + rwd3 + rwd4
        return rwd

    # ------------ _change_cmds functions----------------

    def _change_cmds(self, vx, vy, vang, height):
        # change command_ranges with the input
        self.commands[:, 0] = vx
        self.commands[:, 1] = vy
        self.commands[:, 2] = vang
        self.commands[:, 3] = height

    # ************************ RMA specific ************************
    def reset(self):
        super().reset()
        # self.obs_dict['priv_info'] = self.priv_info_buf.to(self.device)
        # self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device).flatten(1)
        return self.obs_dict



    def _setup_priv_option_config(self, p_config):
        self.enable_priv_enableMeasuredVel = p_config.enableMeasuredVel
        self.enable_priv_measured_height = p_config.enableMeasuredHeight
        self.enable_priv_disturbance_force = p_config.enableForce
        self.enable_priv_Zheights_weights = p_config.enable_priv_Zheights_weights
        self.enable_priv_ZXYheights = p_config.enable_priv_ZXYheights
        self.enable_priv_feet_height = p_config.enable_priv_feet_height
        self.enable_priv_ang_vel = p_config.enable_priv_ang_vel

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.cfg.domain_rand.randomize_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0



    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        # self.prop_hist_len = self.cfg.env.num_histroy_obs
        # self.num_env_priv_obs = self.cfg.env.num_env_priv_obs
        # self.priv_info_buf = torch.zeros((num_envs, self.num_env_priv_obs), device=self.device, dtype=torch.float)
        # self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, self.num_obs), device=self.device,
        #                                     dtype=torch.float)
        super()._allocate_task_buffer(num_envs)

    ############## Dream ########
    def _reward_power_joint(self):
        a = torch.norm(self.torques[:, ], p=1, dim=1)
        b = torch.norm(self.dof_vel[:, ], p=1, dim=1)
        r = a * b
        return r

    def _reward_dream_smoothness(self):
        return torch.sum(torch.square(self.actions - 2 * self.last_actions + self.last_actions_2), dim=1)

    def _reward_power_distribution(self):
        r = torch.mul(self.torques[:, ], self.dof_vel[:, ])
        # d = torch.square(r)

        c = torch.var(r, dim=1)
        d = torch.sum(torch.abs(r), dim=1)
        return d

        # return d

    def _reward_foot_clearance(self):
        self.re_foot = torch.zeros_like(self.foot_height)
        for i in range(0, 4):
            real_foot_height = torch.mean(self.foot_height[:, i].unsqueeze(1) - self.measured_heights, dim=1)
            c = torch.sum(torch.abs(self.foot_vel[:, 2 * i: 2 * i + 2]), dim=1)
            b = torch.norm(self.foot_vel[:, 2 * i: 2 * i + 2], p=1, dim=1)
            self.re_foot[:, i] = torch.mul(torch.square(real_foot_height - self.cfg.rewards.foot_height_target), c)

        foot_reward = torch.sum(self.re_foot, dim=1)

        return foot_reward

    def _reward_foot_height(self):
        self.re_foot = torch.zeros_like(self.foot_height)
        for i in range(0, 4):
            real_foot_height = torch.mean(self.foot_height[:, i].unsqueeze(1) - self.measured_heights, dim=1)

            self.re_foot[:, i] = torch.square(real_foot_height - self.cfg.rewards.foot_height_target)

        # a = torch.square(self.foot_height[:, ])
        b = torch.sum(torch.square(self.re_foot[:, ]), dim=1)
        foot_reward = torch.exp(-10 * b)

        # cprint(f"foot: {self.foot_height, b,  self.re_foot, foot_reward}", 'green', attrs=['bold'])

        return foot_reward


    def _reward_orientation_base(self):
        return torch.sum(torch.square(self.projected_gravity), dim=1)

    ############## RMA ########
    def _reward_RMA_work(self):
        return torch.norm(torch.mul(self.torques, (self.dof_pos - self.last_dof_pos)),  dim=-1)

    def _reward_RMA_ground_impact(self):
        a = self.contact_forces[:, self.feet_indices, :]
        b = self.last_contact_forces[:, self.feet_indices, :]
        f1 = torch.sum(torch.sum(torch.square(a-b), dim=-1), dim=1)
        return f1

    def _reward_RMA_smoothness(self):

        a = self.torques
        b = self.last_torques
        f1 = torch.sum(torch.square(a-b), dim=1)

        return f1
    def _reward_RMA_foot_slip(self):
        g = self.contact_forces[:, self.feet_indices, 2] > 1.
        a = g.long()
        b = self.foot_vel
        f1 = torch.sum(torch.square(torch.mul(a, b)), dim=1)
        return f1

    def _reward_RMA_action_magnitude(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_feet_step(self):
        self.num_calls += 1

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)

        rb_states_3d = rb_states.reshape(self.num_envs, -1, rb_states.shape[-1])
        feet_heights = rb_states_3d[
            :, self.feet_indices, 2
        ]  # proper way to get feet heights, don't use global feet ind
        feet_heights = feet_heights.view(-1)

        xy_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :2], dim=2
        ).view(-1)
        z_forces = self.contact_forces[:, self.feet_indices, 2].view(-1)
        z_forces = torch.abs(z_forces)

        contact = torch.logical_or(
            self.contact_forces[:, self.feet_indices, 2] > 1.0,
            self.contact_forces[:, self.feet_indices, 1] > 1.0,
        )
        contact = torch.logical_or(
            contact, self.contact_forces[:, self.feet_indices, 0] > 1.0
        )

        self.last_contacts = contact
        xy_forces[feet_heights < 0.05] = 0
        z_forces[feet_heights < 0.05] = 0
        z_ans = z_forces.view(-1, 4).sum(dim=1)
        z_ans[z_ans > 1] = 1

        return z_ans


    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        ans = torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

        return ans




    # Eval Functions:
    def _eval_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        ans = torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

        return ans  # (num_robots)

    def _eval_feet_step(self):
        self.num_calls += 1

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)

        rb_states_3d = rb_states.reshape(self.num_envs, -1, rb_states.shape[-1])
        feet_heights = rb_states_3d[
                       :, self.feet_indices, 2
                       ]  # proper way to get feet heights, don't use global feet ind
        feet_heights = feet_heights.view(-1)

        xy_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :2], dim=2
        ).view(-1)
        z_forces = self.contact_forces[:, self.feet_indices, 2].view(-1)
        z_forces = torch.abs(z_forces)

        contact = torch.logical_or(
            self.contact_forces[:, self.feet_indices, 2] > 1.0,
            self.contact_forces[:, self.feet_indices, 1] > 1.0,
        )
        contact = torch.logical_or(
            contact, self.contact_forces[:, self.feet_indices, 0] > 1.0
        )

        contact_filt = torch.logical_or(contact, self.last_contacts).view(-1)
        self.last_contacts = contact
        # print("contact filt shape: ", contact_filt.shape)

        xy_forces[feet_heights < 0.05] = 0
        xy_ans = xy_forces.view(-1, 4).sum(dim=1)

        # print("lowest contacting foot: ", feet_heights[z_forces > 1].min())
        # print("highest contacting foot: ", feet_heights[z_forces > 50.0].max())
        z_forces[feet_heights < 0.05] = 0
        z_ans = z_forces.view(-1, 4).sum(dim=1)
        z_ans[z_ans > 1] = 1  # (num_robots)

        # ans = torch.ones(1024).cuda()
        return z_ans

    def _eval_crash_freq(self):
        reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )

        # print("reset buff: ", reset_buf)

        return reset_buf

    def _eval_any_contacts(self):
        stumble = self._eval_feet_stumble()
        step = self._eval_feet_step()

        return torch.logical_or(stumble, step)
    
def quaternion_multiply(q1, q0):
    """
    Multiply two n * 4 quaternion tensors.
    """
    x0, y0, z0, w0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    return torch.stack([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dim=-1)

def quaternion_inverse(q):
    return torch_normalize(quaternion_conjugate(q))

def quaternion_apply(q, v):
    # q: x, y, z, w
    # v: x, y, z
    q = torch_normalize(q)
    v = torch.cat([v, torch.zeros_like(v[:, 0:1])], dim=-1)
    return quaternion_multiply(q, quaternion_multiply(v, quaternion_conjugate(q)))[:, :3]

def quaternion_conjugate(q):
    # x, y, z, w
    return torch.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], dim=-1)

def torch_normalize(v):
    norm = torch.norm(v, dim=-1, keepdim=True)
    return v / norm
