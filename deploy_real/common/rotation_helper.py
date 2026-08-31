"""Gravity-orientation helper. Identical formula to unitree_rl_gym so sim2real matches."""
import numpy as np


def get_gravity_orientation(quaternion):
    """projected_gravity from a (w,x,y,z) base quaternion. Equals R(q)^T @ [0,0,-1].

    This MUST match the obs used in sim/train: legged_robot.py builds
    projected_gravity = quat_rotate_inverse(base_quat, [0,0,-1]); this closed form
    is the same quantity for a (w,x,y,z) quat.
    """
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g
