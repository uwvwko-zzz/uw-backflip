
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

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from .create_trimesh import QRC_trimesh


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        self.vis_type = cfg.vis_type
        self.jump = cfg.jump
        if self.type in ["none", 'plane']:
            return
        if self.type == "parkour" or "QRC":
            self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
            # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
            self.num_goals = cfg.num_goals
            self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals*9, 3))
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)
        if self.type == "trimesh_no_stair":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

        if self.type == "stair":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)
        if self.type == "stone":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)
        if self.type == "obs_stone":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

        if self.type == "QRC":
            self.vertices_ground, self.triangles_ground = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
            self.vertices = self.vertices_ground
            self.triangles= self.triangles_ground

            goals = self.goals # num_row, num_cols, num_goals, 3
            goals[:,:,:,0] = goals[:,:,:,0] + 2
            self.QRC_vertices, self.QRC_triangles = self.get_QRC_trimeshes(cfg.obj_path, goals) # 
            #print('QRC vertices shape are:',len(self.QRC_vertices))
            #print('QRC triangles shape are:',len(self.QRC_triangles))
            for i in range(len(self.QRC_vertices)):
                # add increment of the triangles
                self.QRC_triangles[i] = self.QRC_triangles[i] + self.vertices.shape[0]
                self.vertices = np.concatenate((self.vertices, self.QRC_vertices[i]), axis=0)
                self.triangles = np.concatenate((self.triangles, self.QRC_triangles[i]), axis=0)
            #print('all_vertices are:',self.vertices)
            #print('all triangles are:',self.triangles)
            print("Created {} vertices (including QRC) ".format(self.vertices.shape[0]))
            print("Created {} triangles (including QRC) ".format(self.triangles.shape[0]))

    def get_QRC_trimeshes(self,obj_path, center_position): # goals: [rows, cols, goals, 3]
        # create mulitple frame trimesh
        frame_vertices, frame_triangles = [], []

        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                for k in range(4*9): #
                    cp = center_position[i, j, k] + np.array([self.cfg.border_size, self.cfg.border_size, 0])
                    frame_vertices_cur, frame_triangles_cur = QRC_trimesh(cp.astype(np.float32),
                                                                          obj_path[0]
                                                                           )
                    # frame_vertices_cur, frame_triangles_cur = box_trimesh(size=np.array([10.1, 10.6, 10.6]),
                    #                                                         center_position=center_position.astype(np.float32)
                    #                                                         )
                    frame_vertices.append(frame_vertices_cur)
                    frame_triangles.append(frame_triangles_cur)
                for k in range(4*9,self.cfg.num_goals*9):
                    cp = center_position[i, j, k] + np.array([self.cfg.border_size, self.cfg.border_size, 0])
                    frame_vertices_cur, frame_triangles_cur = QRC_trimesh(cp.astype(np.float32),
                                                                          obj_path[1]
                                                                           )
                    # frame_vertices_cur, frame_triangles_cur = box_trimesh(size=np.array([10.1, 10.6, 10.6]),
                    #                                                         center_position=center_position.astype(np.float32)
                    #                                                         )
                    frame_vertices.append(frame_vertices_cur)
                    frame_triangles.append(frame_triangles_cur)

        return frame_vertices, frame_triangles
                                                                                    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)

            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j,choice)

    def curiculum(self, diff=None, obs_scale=1):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001
                # choice = 0.21
                terrain = self.make_terrain(choice, difficulty, obs_scale)
                self.add_terrain_to_map(terrain, i, j, choice)



    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.vertical_scale,
                                               horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty,  obs_scale=1):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4  # difficulty = i / self.cfg.num_rows
        step_height = 0.1 + 0.28 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        hurdle_height = 2.0 * difficulty
        gap_size = 0.7 * difficulty
        if difficulty == 0:
            #gap_size = 0.3
            hurdle_height = 0.06
        pit_depth = 1. * difficulty
        if self.type == "stair":
            # terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
            #                                              platform_size=3.)
            #terrain_utils.stairs_terrain(terrain, step_width=1.0, step_height=step_height)
            # parkour_gap_terrain(terrain,
            #                gap_size=gap_size,# should increase with the curriculum
            #                platform_len=1.2, 
            #                platform_height=0., 
            #                num_gaps=6,
            #                x_range=[1.1, 1.5], # the minimum value should larger than the gap_size
            #                y_range=[-0.01, 0.01],
            #                #y_range=[-1.2, 1.2],
            #                half_valid_width=[0.6, 1.2],
            #                gap_depth=-1000,
            #                pad_width=0.1,
            #                pad_height=0.)
            gap_terrain(terrain, gap_size=gap_size, platform_size=3., x_range=[1., 1.4], num_gaps=3)
            #pit_terrain(terrain, depth=pit_depth, platform_size=4.)
            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
            #                                      downsampled_scale=0.2)
            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005,
            #                                          downsampled_scale=0.2)

            # if choice < self.proportions[0]:
            #     # if choice < self.proportions[0] / 2:
            #     #     slope *= -1
            #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
            #                                              platform_size=3.)
            #     # terrain_utils.stairs_terrain(terrain, step_width=0.85, step_height=step_height)
            #     # terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
            #     #                                          platform_size=3.)
            # elif choice < self.proportions[1]:
            #     # terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
            #                                              platform_size=3.)
            #     # terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
            #     #                                      downsampled_scale=0.2)
            #     terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005,
            #                                          downsampled_scale=0.2)
            # elif choice < self.proportions[2]:
            #         # if choice < self.proportions[2]:
            #         #     step_height *= -1
            #         # terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height,
            #         #                                      platform_size=3.)
            #         terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
            #                                              platform_size=3.)
            # elif choice < self.proportions[3]:
            #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
            #                                          platform_size=3.)
            # elif choice < self.proportions[4]:
            #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
            #                                          platform_size=3.)
            # # elif choice < self.proportions[5]:
            # #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
            # #                                          platform_size=3.)
            #     # gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
            # else:
            #     # pit_terrain(terrain, depth=pit_depth, platform_size=4.)
                # terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.85, step_height=step_height,
                #                                      platform_size=3.)

        # elif self.type == "rsl":
        #     slope = difficulty * 0.4
        #     step_height = 0.05 + 0.18 * difficulty
        #     discrete_obstacles_height = 0.05 + difficulty * 0.2
        #     stepping_stones_size = 1.5 * (1.05 - difficulty)
        #     stone_distance = 0.05 if difficulty==0 else 0.1
        #     gap_size = 1. * difficulty
        #     pit_depth = 1. * difficulty
        #     if choice < self.proportions[0]:
        #         if choice < self.proportions[0]/ 2:
        #             slope *= -1
        #         terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        #     elif choice < self.proportions[1]:
        #         terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        #         terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        #     elif choice < self.proportions[3]:
        #         if choice<self.proportions[2]:
        #             step_height *= -1
        #         terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        #     elif choice < self.proportions[4]:
        #         num_rectangles = 20
        #         rectangle_min_size = 1.
        #         rectangle_max_size = 2.
        #         terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        #     elif choice < self.proportions[5]:
        #         terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        #     elif choice < self.proportions[6]:
        #         gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        #     else:
        #         pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        elif self.type == "stone":
            if choice < self.proportions[0]:
                if choice < self.proportions[0] / 2:
                    slope *= -1
                terrain_utils.pyramid_sloped_terrain(
                    terrain, slope=slope, platform_size=3.0
                )
            elif choice < self.proportions[1]:
                terrain_utils.pyramid_sloped_terrain(
                    terrain, slope=slope, platform_size=3.0
                )
                terrain_utils.random_uniform_terrain(
                    terrain,
                    min_height=-0.05,
                    max_height=0.05,
                    step=0.005,
                    downsampled_scale=0.2,
                )
            elif choice < self.proportions[3]:
                if choice < self.proportions[2]:
                    step_height *= -1
                terrain_utils.pyramid_stairs_terrain(
                    terrain, step_width=0.31, step_height=step_height, platform_size=3.0
                )
            elif choice < self.proportions[4]:
                num_rectangles = 20
                rectangle_min_size = 1.0
                rectangle_max_size = 2.0
                terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    discrete_obstacles_height,
                    rectangle_min_size,
                    rectangle_max_size,
                    num_rectangles,
                    platform_size=3.0,
                )
            elif choice < self.proportions[5]:
                # print("MAKING TERRAIN HERE")
                num_rectangles = int(200 * difficulty)
                # num_rectangles = 0
                rectangle_min_size = 2 * obs_scale
                rectangle_max_size = 5 * obs_scale
                terrain_utils.discrete_obstacles_terrain_cells(
                    terrain,
                    # float(os.environ["ISAAC_BLOCK_MIN_HEIGHT"]),
                    # float(os.environ["ISAAC_BLOCK_MAX_HEIGHT"]),
                    0.14,
                    0.15,
                    rectangle_min_size,
                    rectangle_max_size,
                    num_rectangles,
                    platform_size=3.0,
                    width=2 * obs_scale
                )
            elif choice < self.proportions[6]:
                terrain_utils.stepping_stones_terrain(
                    terrain,
                    stone_size=stepping_stones_size,
                    stone_distance=stone_distance,
                    max_height=0.0,
                    platform_size=4.0,
                )
            elif choice < self.proportions[7]:
                gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
            else:
                pit_terrain(terrain, depth=pit_depth, platform_size=4.0)
        elif self.type == "obs_stone":
            if choice < self.proportions[0]:
                if choice < self.proportions[0] / 2:
                    slope *= -1
                terrain_utils.pyramid_sloped_terrain(
                    terrain, slope=slope, platform_size=3.0
                )
            elif choice < self.proportions[1]:
                terrain_utils.pyramid_sloped_terrain(
                    terrain, slope=slope, platform_size=3.0
                )
                terrain_utils.random_uniform_terrain(
                    terrain,
                    min_height=-0.05,
                    max_height=0.05,
                    step=0.005,
                    downsampled_scale=0.2,
                )
            elif choice < self.proportions[3]:
                if choice < self.proportions[2]:
                    step_height *= -1
                terrain_utils.pyramid_stairs_terrain(
                    terrain, step_width=0.31, step_height=step_height, platform_size=3.0
                )
            elif choice < self.proportions[4]:
                num_rectangles = 20
                rectangle_min_size = 1.0
                rectangle_max_size = 2.0
                terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    discrete_obstacles_height,
                    rectangle_min_size,
                    rectangle_max_size,
                    num_rectangles,
                    platform_size=3.0,
                )
            # elif choice < self.proportions[5]:
            #     # print("MAKING TERRAIN HERE")
            #     num_rectangles = int(200 * difficulty)
            #     # num_rectangles = 0
            #     rectangle_min_size = 2 * obs_scale
            #     rectangle_max_size = 5 * obs_scale
            #     add_terrain_utils.discrete_obstacles_terrain_cells(
            #         terrain,
            #         0.14,
            #         0.15,
            #         rectangle_min_size,
            #         rectangle_max_size,
            #         num_rectangles,
            #         platform_size=3.0,
            #         width=2 * obs_scale
            #     )
            # elif choice < self.proportions[6]:
                terrain_utils.stepping_stones_terrain(
                    terrain,
                    stone_size=stepping_stones_size,
                    stone_distance=stone_distance,
                    max_height=0.0,
                    platform_size=4.0,
                )
            elif choice < self.proportions[7]:
                gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
            else:
                pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        
        elif self.type == "QRC":
            # QRC_terrain(terrain,
            #             platform_len=2.5,
            #                 platform_height=0.,
            #                 num_stones=self.num_goals-2,
            #                 #x_range=[1.0, 3.0],
            #                 #y_range=self.cfg.y_range,
            #                 #frame_height = self.cfg.frame_height,
            #                 #hurdle_height_range=self.cfg.center_position_z,
            #                )
            # idx = 0
            # terrain.idx = idx 
            if choice < self.proportions[0]:
                if choice < self.proportions[0] / 2:
                    slope *= -1
                terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                idx = 0
                terrain.idx = idx   
            elif choice < self.proportions[1]:
                terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                     downsampled_scale=0.2)
                idx = 1
                terrain.idx = idx

            elif choice < self.proportions[2]:
                num_rectangles = 20
                rectangle_min_size = 1.
                rectangle_max_size = 2.
                # if self.type == "trimesh_no_stair":
                #     pass
                # else:
                #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=2.)
                #     terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                #                                          downsampled_scale=0.2)
                terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
                idx = 2
                terrain.idx = idx 
 
            elif choice < self.proportions[3]:
                QRC_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=self.num_goals-2,
                           x_range=[1.2, 1.4],
                           y_range=[0.0, 0.02],
                           frame_height=0,
                           hurdle_height_range=[0.08, 0.10], #[0.1,0.14]
                           )
                idx = 3
                terrain.idx = idx 
            
            elif choice < self.proportions[4]:
                QRC_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=self.num_goals-2,
                           x_range=[1.2, 1.4],
                           y_range=[0.0, 0.02],
                           frame_height=0,
                           hurdle_height_range=[0.10, 0.14], #[0.1,0.14]
                           )
                idx = 4
                terrain.idx = idx  

            elif choice < self.proportions[5]:
                QRC_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=self.num_goals-2,
                           x_range=[1.2, 1.4],
                           y_range=[0.0, 0.02],
                           frame_height=0,
                           hurdle_height_range=[0.14, 0.16], #[0.1,0.14]
                           )
                idx = 5
                terrain.idx = idx                          
            elif choice < self.proportions[6]:
                QRC_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=self.num_goals-2,
                           x_range=[1.2, 1.4],
                           y_range=[0.0, 0.02],
                           frame_height=0,
                           hurdle_height_range=[0.16, 0.20], #[0.1,0.14]
                           )
                idx = 6
                terrain.idx = idx 
            else:
                pit_terrain(terrain, depth=pit_depth, platform_size=4.)
                idx = 7
                terrain.idx = idx 

      

        else:
            # normal terrain for 'trimesh'
            # if choice < self.proportions[0]:
            #     if choice < self.proportions[0] / 2:
            #         slope *= -1
            #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # elif choice < self.proportions[1]:
            #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            #     terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
            #                                          downsampled_scale=0.2)
            # elif choice < self.proportions[3]:
            #     if self.type == "trimesh_no_stair":
            #         terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            #         terrain_utils.random_uniform_terrain(terrain, min_height=-0.01, max_height=0.01, step=0.005,
            #                                              downsampled_scale=0.2)
            #     else:
            #         if choice < self.proportions[2]:
            #             step_height *= -1
            #         terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)  

            # if choice < self.proportions[0]:
            #     num_rectangles = 20
            #     rectangle_min_size = 1.
            #     rectangle_max_size = 2.
            #     terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
            #                                          rectangle_max_size, num_rectangles, platform_size=3.)
            # elif choice < self.proportions[1]:
            #     terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
            #                                           stone_distance=stone_distance, max_height=0., platform_size=4.)
            # elif choice < self.proportions[2]:
            #     gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
            # elif choice < self.proportions[3]:
            #     parkour_hurdle_terrain(terrain, gap_size=0.5,gap_depth=0.3)
            # else:
            #     pit_terrain(terrain, depth=pit_depth, platform_size=4.)

            # flat ground:
            if self.vis_type == 'train':
                QRC_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=self.num_goals-2,
                           x_range=[1.2, 1.4],
                           y_range=[0.0, 0.02],
                           frame_height=0,
                           hurdle_height_range=[0.08, 0.10], #[0.1,0.14]
                           )
                #self.add_roughness(terrain)
            #jump_rectangle_terrain(terrain, hurdle_height_range=[hurdle_height, hurdle_height])

            # jumping terrain
            elif self.vis_type == 'test':
                if choice < self.proportions[0]:
                    num_rectangles = 30
                    rectangle_min_size = 1.
                    rectangle_max_size = 2.
                    terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    0.12,
                    rectangle_min_size,
                    rectangle_max_size,
                    num_rectangles,
                    platform_size=3.0,
                )
                #     terrain_utils.pyramid_stairs_terrain(
                #     terrain, step_width=0.31, step_height=0.15, platform_size=3.0
                # )
                    #parkour_hurdle_terrain(terrain, gap_size=0.2,gap_depth=0.25)
                    idx = 0
                    terrain.idx = idx   
                elif choice < self.proportions[1]:
                    num_rectangles = 30
                    rectangle_min_size = 1.
                    rectangle_max_size = 2.
                    terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    0.1,
                    rectangle_min_size,
                    rectangle_max_size,
                    num_rectangles,
                    platform_size=3.0,
                )
                #     terrain_utils.pyramid_stairs_terrain(
                #     terrain, step_width=0.31, step_height=0.15, platform_size=3.0
                # )
                    #parkour_hurdle_terrain(terrain, gap_size=0.2,gap_depth=0.25)
                    idx = 1
                    terrain.idx = idx 
                else:
                    num_rectangles = 40
                    rectangle_min_size = 1.
                    rectangle_max_size = 2.
                #     terrain_utils.pyramid_stairs_terrain(
                #     terrain, step_width=0.31, step_height=0.15, platform_size=3.0
                # )
                    terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    0.11,
                    rectangle_min_size,
                    rectangle_max_size,
                    num_rectangles,
                    platform_size=3.0,
                )
                    #parkour_hurdle_terrain(terrain, gap_size=0.2, gap_depth=0.2)
                    idx = 2
                    terrain.idx = idx 
            
            # else:
            #     parkour_gap_terrain(terrain, gap_size=gap_size)
            #     idx = 2
            #     terrain.idx = idx
            
        
        #self.add_roughness(terrain)
        return terrain




    def add_terrain_to_map(self, terrain, row, col, choice):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw
        # add the height value to the QRC map..

        if self.type == "QRC":
            env_origin_x = i * self.env_length + 1.0
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 0.5) / terrain.horizontal_scale)  # within 1 meter square range
            x2 = int((self.env_length / 2. + 0.5) / terrain.horizontal_scale)
            y1 = int((self.env_width / 2. - 0.5) / terrain.horizontal_scale)
            y2 = int((self.env_width / 2. + 0.5) / terrain.horizontal_scale)
            if self.cfg.origin_zero_z:
                env_origin_z = 0
            else:
                env_origin_z = self.cfg.init_state.pos[2]#np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
            self.terrain_type[i, j] = terrain.idx
            if terrain.idx in [3,4,5,6]:
                self.goals[i, j, :, :] = terrain.goals + [i * self.env_length, j * self.env_width, 0]
        else:
            #env_origin_x = (i + 0.15) * self.env_length
            env_origin_x = (i + 0.35) * self.env_length
            env_origin_y = (j + 0.35) * self.env_width
            x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
            if self.cfg.origin_zero_z:
                env_origin_z = 0
            else:
                env_origin_z =  (np.max(terrain.height_field_raw[x1:x2, y1:y2])) * terrain.vertical_scale
                #env_origin_z =  terrain.height_field_raw[x1:x2, y1:y2] * terrain.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]            
            # env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            # self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
            #if self.jump:
            #    self.goals[i, j, :, :] = terrain.goals + [i * self.env_length, j * self.env_width]
def gap_terrain(terrain, gap_size, platform_size=1.,x_range=[1.0, 1.2], y_range=[-0.01, 0.01], num_gaps=4):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0

    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + 2*gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + 2*gap_size

    for i in range(num_gaps-1):
        dis_x = np.random.randint(dis_x_min, dis_x_max)
        x1 = x1 - dis_x
        x2 = x1 + gap_size
        y1 = y1 - dis_x
        y2 = y1 + gap_size
        terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
        terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0    
def parkour_gap_terrain(terrain,
                           gap_size=1.6,# should increase with the curriculum
                           platform_len=1.8, 
                           platform_height=0., 
                           num_gaps=5,
                           x_range=[1.2, 1.8], # the minimum value should larger than the gap_size
                           y_range=[-0.01, 0.01],
                           #y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-1000,
                           pad_width=0.1,
                           pad_height=0.,#0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))          # goals are used to define the waypoint on the road.
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = 0#round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = 0.1#round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale) # means how many grids in total.
    platform_height = round(platform_height / terrain.vertical_scale)
    #gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale) # vartival_scale=0.005 
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height # only descibe the height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len # the starting position of the first gap
    goals[0] = [platform_len - 1, mid_y]  # The first dimension of goals is 
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max) #(dis_x_min+ dis_x_max)//2
        dis_x += rand_x # the distance between 2 gaps
        rand_y = 0#np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        #terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth  # dist_x is a random value
        #terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    terrain.goals = goals * terrain.horizontal_scale

def vertical_parkour_gap_terrain(terrain,
                           gap_size=1.6,# should increase with the curriculum
                           platform_len=1.8, 
                           platform_height=0., 
                           num_gaps=5,
                           x_range=[1.2, 1.8], # the minimum value should larger than the gap_size
                           y_range=[-0.01, 0.01],
                           #y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-1000,
                           pad_width=0.1,
                           pad_height=0.,#0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))          # goals are used to define the waypoint on the road.
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = 0#round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = 0.1#round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale) # means how many grids in total.
    platform_height = round(platform_height / terrain.vertical_scale)
    #gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale) # vartival_scale=0.005 
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height # only descibe the height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len # the starting position of the first gap
    goals[0] = [platform_len - 1, mid_y]  # The first dimension of goals is 
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max) #(dis_x_min+ dis_x_max)//2
        dis_x += rand_x # the distance between 2 gaps
        rand_y = 0#np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        #terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth  # dist_x is a random value
        #terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    terrain.goals = goals * terrain.horizontal_scale

def parkour_hurdle_terrain(terrain,
                           gap_size=0.4,# should increase with the curriculum
                           gap_depth=100,
                           platform_len=1.8, 
                           platform_height=0., 
                           num_gaps=2,
                           x_range=[1.2, 1.8], # the minimum value should larger than the gap_size
                           y_range=[-0.01, 0.01],
                           #y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           pad_width=0.1,
                           pad_height=0.,#0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))          # goals are used to define the waypoint on the road.
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = 0#round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = 0.1#round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale) # means how many grids in total.
    platform_height = round(platform_height / terrain.vertical_scale)
    #gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale) # vartival_scale=0.005 
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height # only descibe the height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len*2
    goals[0] = [platform_len - 1, mid_y]  # The first dimension of goals is 

    for i in range(num_gaps):
        rand_x = (dis_x_min+ dis_x_max)//2#np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = 0#np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = (gap_depth+i*0.05)/ terrain.vertical_scale 

        #terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth  # dist_x is a random value
        #terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    terrain.goals = goals * terrain.horizontal_scale

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def QRC_terrain(terrain,
                           platform_len=4.0,
                           platform_height=0.,
                           num_stones=6, # goals -2
                           x_range=[1.2, 1.4],
                           y_range=[0.0, 0.02],
                           frame_height=0,
                           hurdle_height_range=[0.16, 0.20], #[0.1,0.14]
                           ):
    #goals = np.zeros((num_stones+2, 3))  # (num_goals, 2); 2 for x-y
    goals = np.zeros((9*(num_stones+2), 3))
    #reward_point = np.zeros((num_stones, 3))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    #dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    #dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    rand_y = (y_range[0] / terrain.horizontal_scale + y_range[1] / terrain.horizontal_scale)/2

    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    #steps = (hurdle_height_max-hurdle_height_min)/num_stones
    #hurdle_height_list = np.arange(hurdle_height_min, hurdle_height_max, steps)

    # wall_width=4
    # start2center=0.7
    # max_height=np.random.randint(20, 30)
    # wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    # max_height_int = int(max_height / terrain.vertical_scale)

    # terrain_length = terrain.length
    # height2width_ratio = max_height_int / wall_width_int

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y, hurdle_height_min]
    goals[1] = [platform_len - 1, mid_y+mid_y/4, hurdle_height_min]
    goals[2] = [platform_len - 1, mid_y-mid_y/4, hurdle_height_min]
    goals[3] = [platform_len - 1, mid_y+mid_y/2, hurdle_height_min]
    goals[4] = [platform_len - 1, mid_y-mid_y/2, hurdle_height_min]
    goals[5] = [platform_len - 1, mid_y-mid_y+7, hurdle_height_min]
    goals[6] = [platform_len - 1, mid_y+mid_y-7, hurdle_height_min]
    goals[7] = [platform_len - 1, mid_y-mid_y+18.5, hurdle_height_min]
    goals[8] = [platform_len - 1, mid_y+mid_y-18.5, hurdle_height_min]


    reward_point = np.zeros((num_stones, 3))

    # second lateral and diagonal:
    for i in range(6):
        #hurdle_height = hurdle_height_list[i]
        hurdle_height = np.random.randint(hurdle_height_min, hurdle_height_max)
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        #rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x

        goals[9*i+9] = [dis_x, mid_y + rand_y,hurdle_height]
        goals[9*i+10] = [dis_x, mid_y + rand_y+mid_y/2,hurdle_height]
        goals[9*i+11] = [dis_x, mid_y + rand_y-mid_y/2,hurdle_height]
        goals[9*i+12] = [dis_x, mid_y + rand_y+mid_y/4,hurdle_height]
        goals[9*i+13] = [dis_x, mid_y + rand_y-mid_y/4,hurdle_height]
        goals[9*i+14] = [dis_x, mid_y + mid_y-7,hurdle_height]
        goals[9*i+15] = [dis_x, mid_y -mid_y+7,hurdle_height]
        goals[9*i+16] = [dis_x, mid_y + mid_y-18.5,hurdle_height]
        goals[9*i+17] = [dis_x, mid_y -mid_y+18.5,hurdle_height]


        reward_point[i] = [dis_x, mid_y+rand_y, 2 * hurdle_height]

    for i in range(6, num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        # slope_start = int(dis_x-rand_x)
        # xs = np.arange(slope_start, terrain_length)
        # heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
        # terrain.height_field_raw[slope_start:xs, :] = 20#heights[:, None]
        # terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
        # terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)



        # hurdle_height = max_height/2
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        #rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x

        goals[9*i+9] = [dis_x, mid_y + rand_y,hurdle_height]
        goals[9*i+10] = [dis_x, mid_y + rand_y+mid_y/2,hurdle_height]
        goals[9*i+11] = [dis_x, mid_y + rand_y-mid_y/2,hurdle_height]
        goals[9*i+12] = [dis_x, mid_y + rand_y+mid_y/4,hurdle_height]
        goals[9*i+13] = [dis_x, mid_y + rand_y-mid_y/4,hurdle_height]
        goals[9*i+14] = [dis_x, mid_y + mid_y-7,hurdle_height]
        goals[9*i+15] = [dis_x, mid_y -mid_y+7,hurdle_height]
        goals[9*i+16] = [dis_x, mid_y + mid_y-18.5,hurdle_height]
        goals[9*i+17] = [dis_x, mid_y -mid_y+18.5,hurdle_height]

        reward_point[i] = [dis_x, mid_y+rand_y, 2 * hurdle_height]

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y + rand_y,hurdle_height_max]
    goals[(num_stones+2)*9-7] = [final_dis_x, mid_y+rand_y+mid_y/2,hurdle_height_max]
    goals[(num_stones+2)*9-6] = [final_dis_x, mid_y+rand_y-mid_y/2,hurdle_height_max]
    goals[(num_stones+2)*9-5] = [final_dis_x, mid_y+rand_y+mid_y/4,hurdle_height_max]
    goals[(num_stones+2)*9-4] = [final_dis_x, mid_y+rand_y-mid_y/4,hurdle_height_max]
    goals[(num_stones+2)*9-3] = [final_dis_x, mid_y-mid_y+7,hurdle_height_max]
    goals[(num_stones+2)*9-2] = [final_dis_x, mid_y+mid_y-7,hurdle_height_max]
    goals[(num_stones+2)*9-9] = [final_dis_x, mid_y-mid_y+18.5,hurdle_height_max]
    goals[(num_stones+2)*9-8] = [final_dis_x, mid_y+mid_y-18.5,hurdle_height_max]


    goals[:,0:2] *= terrain.horizontal_scale
    goals[:,2] *= terrain.vertical_scale
    terrain.goals = goals
    reward_point[:,0:2] *= terrain.horizontal_scale
    reward_point[:,2] *= terrain.vertical_scale
    terrain.reward_point = reward_point

    # max_height = terrain.height_field_raw.max()
    # top_mask = terrain.height_field_raw > max_height - 0.05
    # terrain.height_field_raw[top_mask] = max_height
