import numpy as np
from numpy.linalg import norm, inv
from math import acos, atan2, sqrt, pi
from scipy.spatial.transform import Rotation as R


class Aliengo:
    dist_hip_x = 0.2399
    dist_hip_y = 0.051
    len_hip = 0.083;
    len_thigh = 0.25;
    len_calf = 0.25;



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


    def compute_inverse(self,foot_xyz_B, foot_vel_B):
        #-------calculate calf using variant type of question 2

        foot2hip_B = foot_xyz_B - self.hip_xyz_B
        foot2hip_len = norm(foot2hip_B)
        if foot2hip_len> self.ws_radius:
            print('Warning: foot position %s is out of workspace'%repr(foot_xyz_B.tolist()))
            foot_xyz_B = self.hip_xyz_B + foot2hip_B/foot2hip_len*self.ws_radius
            foot2hip_len = self.ws_radius
            print('\t changed to %s\n'%repr(foot_xyz_B.tolist()))
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

        #rerange by definition
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
        hip = atan2(np.dot(np.cross(u_,v_),w),np.dot(u_,v_))
        
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
        self.ik = [IK(robot,'LF'), IK(robot,'RF'), IK(robot,'LH'), IK(robot,'RH')]

    def computeIK(self, foot_pos, foot_vel):
        joint_pos = np.zeros(12)
        joint_vel = np.zeros(12)
        for i in range(4):
            foot_pos_i = foot_pos[3*i:3*(i+1)]
            foot_vel_i = foot_vel[3*i:3*(i+1)]
            joint_pos_i, joint_vel_i = self.ik[i].compute_inverse(foot_pos_i, foot_vel_i)
            joint_pos[3*i:3*(i+1)] = joint_pos_i  # hip, calf, thigh
            joint_vel[3*i:3*(i+1)] = joint_vel_i  # hip, calf, thigh
        return joint_pos, joint_vel
