from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import nn
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .go2_config_baseline import Go2BaseCfg



class Go2(LeggedRobot):
    cfg: Go2BaseCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)



    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

    def _init_buffers(self):
        super()._init_buffers()

    def _compute_poses(self, actions):
        return super()._compute_poses(actions)

    def _compute_torques(self, actions):
        return super()._compute_torques(actions)





