import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import normalize, quat_from_angle_axis, quat_rotate_inverse

from legged_gym.envs.go2.go2 import Go2


class Go2Backflip(Go2):
    """Minimal phase-conditioned backflip task ported from Genesis-backflip."""

    def _head_body_names(self):
        """Return rigid bodies used by head-clearance/contact rewards."""
        return [name for name in self.body_names if "Head" in name]

    def _init_custom_buffers__(self):
        super()._init_custom_buffers__()

        self._limb_body_indices = [
            index
            for index, name in enumerate(self.body_names)
            if any(part in name for part in ("hip", "thigh", "calf", "foot"))
        ]
        if len(self._limb_body_indices) != 16:
            raise RuntimeError(
                f"Expected 16 Go2 limb bodies, got {len(self._limb_body_indices)}"
            )
        self._limb_slot = {
            body_index: slot
            for slot, body_index in enumerate(self._limb_body_indices)
        }

        mass_range = self.cfg.domain_rand.limb_mass_scale_range
        inertia_range = self.cfg.domain_rand.limb_inertia_scale_range
        self._limb_mass_scales_np = (
            np.random.uniform(mass_range[0], mass_range[1], (self.num_envs, 16))
            if self.cfg.domain_rand.randomize_limb_mass
            else np.ones((self.num_envs, 16))
        )
        inertia_jitter = (
            np.random.uniform(inertia_range[0], inertia_range[1], (self.num_envs, 16))
            if self.cfg.domain_rand.randomize_limb_inertia
            else np.ones((self.num_envs, 16))
        )
        # Inertia follows the sampled mass and receives an additional modeling
        # error multiplier. This is the total factor exposed to the critic.
        self._limb_inertia_scales_np = self._limb_mass_scales_np * inertia_jitter
        self.limb_mass_scales = torch.as_tensor(
            self._limb_mass_scales_np, dtype=torch.float, device=self.device
        )
        self.limb_inertia_scales = torch.as_tensor(
            self._limb_inertia_scales_np, dtype=torch.float, device=self.device
        )

        restitution_range = self.cfg.domain_rand.restitution_range
        contact_offset_range = self.cfg.domain_rand.contact_offset_range
        rest_offset_range = self.cfg.domain_rand.rest_offset_range
        if self.cfg.domain_rand.randomize_contact:
            self._contact_restitution_np = np.random.uniform(
                restitution_range[0], restitution_range[1], self.num_envs
            )
            self._contact_offset_np = np.random.uniform(
                contact_offset_range[0], contact_offset_range[1], self.num_envs
            )
            self._rest_offset_np = np.random.uniform(
                rest_offset_range[0], rest_offset_range[1], self.num_envs
            )
        else:
            self._contact_restitution_np = np.zeros(self.num_envs)
            self._contact_offset_np = np.full(self.num_envs, 0.01)
            self._rest_offset_np = np.zeros(self.num_envs)
        self.contact_restitution = torch.as_tensor(
            self._contact_restitution_np[:, None], dtype=torch.float, device=self.device
        )
        self.contact_offset = torch.as_tensor(
            self._contact_offset_np[:, None], dtype=torch.float, device=self.device
        )
        self.contact_friction = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.base_mass_values = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.base_com_values = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        self.torque_scales = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device
        )
        self.motor_velocity_scales = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device
        )
        self.joint_velocity_limits = torch.tensor(
            self.cfg.control.joint_velocity_limits,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        self.slew_limited_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device
        )
        self.obs_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        max_obs_delay = self.cfg.control.max_observation_delay_steps
        self.sensor_obs_history = torch.zeros(
            max_obs_delay + 1,
            self.num_envs,
            30,
            dtype=torch.float,
            device=self.device,
        )
        self.obs_delay_needs_fill = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.ang_vel_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.gravity_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.dof_pos_bias = torch.zeros(self.num_envs, 12, device=self.device)
        self.dof_vel_bias = torch.zeros(self.num_envs, 12, device=self.device)
        self.flip_angle = torch.zeros(self.num_envs, device=self.device)
        self.max_flip_angle = torch.zeros(self.num_envs, device=self.device)
        self.rotation_progress_step = torch.zeros(self.num_envs, device=self.device)
        self.flip_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.flip_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.just_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.just_succeeded = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @staticmethod
    def _scale_inertia(inertia, scale):
        for row_name in ("x", "y", "z"):
            row = getattr(inertia, row_name)
            row.x *= scale
            row.y *= scale
            row.z *= scale

    def _process_rigid_shape_props(self, props, env_id):
        props = super()._process_rigid_shape_props(props, env_id)
        if self.cfg.domain_rand.randomize_contact:
            restitution = float(self._contact_restitution_np[env_id])
            contact_offset = float(self._contact_offset_np[env_id])
            rest_offset = float(self._rest_offset_np[env_id])
            rest_offset = min(rest_offset, contact_offset - 1.0e-4)
            for shape in props:
                shape.restitution = restitution
                shape.contact_offset = contact_offset
                shape.rest_offset = rest_offset
        self.contact_friction[env_id, 0] = float(props[0].friction)
        return props

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Base uses ceil(), but PhysX exposes 4 * 0.005 as
        # 0.01999999955 and would add an unintended extra control step.
        self.max_episode_length = int(round(self.max_episode_length_s / self.dt))
        self._dof_index = {name: i for i, name in enumerate(self.dof_names)}
        self.rear_dof_indices = torch.tensor(
            [
                self._dof_index[name]
                for name in (
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                )
            ],
            dtype=torch.long,
            device=self.device,
        )
        head_names = self._head_body_names()
        if not head_names:
            raise RuntimeError("Go2 URDF does not expose Head_upper/Head_lower bodies")
        self.head_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in head_names
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _process_rigid_body_props(self, props, env_id):
        original_base_mass = props[0].mass
        props = super()._process_rigid_body_props(props, env_id)
        if self.cfg.domain_rand.randomize_center:
            low, high = self.cfg.domain_rand.added_center_range
            props[0].com.x += np.random.uniform(low, high)
            props[0].com.y += np.random.uniform(low, high)
            props[0].com.z += np.random.uniform(low, high)

        # When recompute_inertia=False, keep base inertia physically consistent
        # with the randomized mass and preserve the explicit limb perturbations.
        self._scale_inertia(props[0].inertia, props[0].mass / original_base_mass)
        for body_index in self._limb_body_indices:
            slot = self._limb_slot[body_index]
            mass_scale = (
                self._limb_mass_scales_np[env_id, slot]
                if self.cfg.domain_rand.randomize_limb_mass
                else 1.0
            )
            inertia_scale = (
                self._limb_inertia_scales_np[env_id, slot]
                if self.cfg.domain_rand.randomize_limb_inertia
                else mass_scale
            )
            props[body_index].mass *= mass_scale
            self._scale_inertia(props[body_index].inertia, inertia_scale)

        self.base_mass_values[env_id, 0] = props[0].mass
        self.base_com_values[env_id] = torch.tensor(
            [props[0].com.x, props[0].com.y, props[0].com.z],
            dtype=torch.float,
            device=self.device,
        )
        return props

    def reset_idx(self, env_ids):
        finished_angle = self.max_flip_angle[env_ids].clone()
        finished_success = self.flip_success[env_ids].float().clone()
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.extras["episode"]["flip_angle_rad"] = torch.mean(finished_angle)
        self.extras["episode"]["flip_success"] = torch.mean(finished_success)
        self.flip_angle[env_ids] = 0.0
        self.max_flip_angle[env_ids] = 0.0
        self.rotation_progress_step[env_ids] = 0.0
        self.flip_completed[env_ids] = False
        self.flip_success[env_ids] = False
        self.just_completed[env_ids] = False
        self.just_succeeded[env_ids] = False

        if self.cfg.domain_rand.randomize_torque_scale:
            low, high = self.cfg.domain_rand.torque_scale_range
            self.torque_scales[env_ids] = low + (high - low) * torch.rand(
                len(env_ids), self.num_actions, device=self.device
            )
        else:
            self.torque_scales[env_ids] = 1.0
        if self.cfg.domain_rand.randomize_motor_velocity:
            low, high = self.cfg.domain_rand.motor_velocity_scale_range
            self.motor_velocity_scales[env_ids] = low + (high - low) * torch.rand(
                len(env_ids), self.num_actions, device=self.device
            )
        else:
            self.motor_velocity_scales[env_ids] = 1.0
        self.slew_limited_actions[env_ids] = 0.0

        max_obs_delay = self.cfg.control.max_observation_delay_steps
        self.obs_delay_steps[env_ids] = torch.randint(
            0, max_obs_delay + 1, (len(env_ids),), device=self.device
        )
        self.sensor_obs_history[:, env_ids] = 0.0
        self.obs_delay_needs_fill[env_ids] = True

        if self.cfg.noise.add_noise:
            for buffer, range_name in (
                (self.ang_vel_bias, "ang_vel_bias_range"),
                (self.gravity_bias, "gravity_bias_range"),
                (self.dof_pos_bias, "dof_pos_bias_range"),
                (self.dof_vel_bias, "dof_vel_bias_range"),
            ):
                low, high = getattr(self.cfg.noise, range_name)
                buffer[env_ids] = low + (high - low) * torch.rand(
                    len(env_ids), buffer.shape[1], device=self.device
                )
        else:
            self.ang_vel_bias[env_ids] = 0.0
            self.gravity_bias[env_ids] = 0.0
            self.dof_pos_bias[env_ids] = 0.0
            self.dof_vel_bias[env_ids] = 0.0

        nominal_p = torch.zeros(self.num_actions, device=self.device)
        nominal_d = torch.zeros(self.num_actions, device=self.device)
        for joint, name in enumerate(self.dof_names):
            for gain_name, stiffness in self.cfg.control.stiffness.items():
                if gain_name in name:
                    nominal_p[joint] = stiffness
                    nominal_d[joint] = self.cfg.control.damping[gain_name]
                    break

        if self.cfg.domain_rand.randomize_kp_scale:
            low, high = self.cfg.domain_rand.kp_scale_range
            kp_scale = low + (high - low) * torch.rand(
                len(env_ids), self.num_actions, device=self.device
            )
        else:
            kp_scale = torch.ones(len(env_ids), self.num_actions, device=self.device)
        if self.cfg.domain_rand.randomize_kd_scale:
            low, high = self.cfg.domain_rand.kd_scale_range
            kd_scale = low + (high - low) * torch.rand(
                len(env_ids), self.num_actions, device=self.device
            )
        else:
            kd_scale = torch.ones(len(env_ids), self.num_actions, device=self.device)

        self.p_gains[env_ids] = nominal_p * kp_scale
        self.d_gains[env_ids] = nominal_d * kd_scale
        if self.cfg.domain_rand.randomize_Motor_Offset:
            low, high = self.cfg.domain_rand.added_Motor_OffsetRange
            self.motor_offsets[env_ids] = low + (high - low) * torch.rand(
                len(env_ids), self.num_actions, device=self.device
            )
        else:
            self.motor_offsets[env_ids] = 0.0

        if "episode" in self.extras:
            self.extras["episode"]["safety_curriculum"] = torch.tensor(
                self._safety_curriculum_scale(), device=self.device
            )
            self.extras["episode"]["velocity_termination_ratio"] = torch.tensor(
                getattr(
                    self,
                    "velocity_termination_ratio",
                    self.cfg.control.velocity_termination_ratio,
                ),
                device=self.device,
            )
    def _compute_torques(self, actions):
        # The shared buffer is rolled once per 50-Hz policy step. Each
        # environment samples index 0/1/2 on reset: 0/20/40 ms delay.
        delayed_actions = self.action_buffer[
            self.env_delay_steps,
            torch.arange(self.num_envs, device=self.device),
        ]
        # Limit the PD target slew rate at every 5-ms physics step. This keeps
        # the policy from hiding an unrealistically large impulse in a one-step
        # target jump while retaining the original 50-Hz actor interface.
        max_action_delta = (
            self.cfg.control.target_velocity_limit
            * self.motor_velocity_scales
            * self.sim_params.dt
            / self.cfg.control.action_scale
        )
        action_delta = torch.clamp(
            delayed_actions - self.slew_limited_actions,
            min=-max_action_delta,
            max=max_action_delta,
        )
        self.slew_limited_actions += action_delta
        actions_scaled = self.slew_limited_actions * self.cfg.control.action_scale
        self.raw_joint_pos_target = (
            delayed_actions * self.cfg.control.action_scale
            + self.default_dof_pos
            + self.motor_offsets
        )
        self.joint_pos_target = (
            actions_scaled + self.default_dof_pos + self.motor_offsets
        )
        torques = (
            self.p_gains * (self.joint_pos_target - self.dof_pos)
            - self.d_gains * self.dof_vel
        )
        # Unitree Go2HV torque-speed envelope. Driving and braking use the
        # official asymmetric 20.2/23.4-Nm peaks; both decay from X1 to X2.
        velocity_scale = self.motor_velocity_scales
        x1 = self.cfg.control.motor_velocity_x1 * velocity_scale
        x2 = self.cfg.control.motor_velocity_x2 * velocity_scale
        speed = torch.abs(self.dof_vel)
        speed_fraction = torch.where(
            speed < x1,
            torch.ones_like(speed),
            torch.clamp((x2 - speed) / torch.clamp(x2 - x1, min=1.0e-6), 0.0, 1.0),
        )
        same_direction = self.dof_vel * torques > 0.0
        peak_torque = torch.where(
            same_direction,
            torch.full_like(torques, self.cfg.control.motor_torque_y1),
            torch.full_like(torques, self.cfg.control.motor_torque_y2),
        ) * self.torque_scales
        torque_limit = peak_torque * speed_fraction
        return torch.clamp(torques, min=-torque_limit, max=torque_limit)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        ids = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(ids), len(ids)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        self.initial_root_states[env_ids] = self.root_states[env_ids]
        ids = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids), len(ids)
        )

    def post_physics_step(self):
        # Never leak episode/time-out information from a previous control step.
        self.extras = {}
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.base_quat[:] = self.root_states[:, 3:7]
        yaw_quat = self.base_quat.clone()
        yaw_quat[:, :2] = 0.0
        yaw_quat = normalize(yaw_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(yaw_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.feet_pos[:] = self.rigid_body_state[:, self.feet_indices, 0:3]
        # Integrate signed backward pitch and reward only a new maximum. This
        # prevents forward/backward oscillation from being counted as a flip.
        previous_max = self.max_flip_angle.clone()
        pitch_rate = torch.clamp(
            -self.base_ang_vel[:, 1],
            min=-self.cfg.rewards.flip_angle_rate_clip,
            max=self.cfg.rewards.flip_angle_rate_clip,
        )
        self.flip_angle += pitch_rate * self.dt
        self.max_flip_angle = torch.maximum(self.max_flip_angle, self.flip_angle)
        self.rotation_progress_step = torch.clamp(
            self.max_flip_angle - previous_max, min=0.0
        )
        self.just_completed = (
            (~self.flip_completed)
            & (self.max_flip_angle >= self.cfg.rewards.flip_completion_angle)
        )
        self.flip_completed |= self.just_completed
        feet_contact_count = torch.sum(
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
            > self.cfg.rewards.recovery_contact_force,
            dim=1,
        )
        pose_error = torch.max(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        ).values
        success_now = (
            self.flip_completed
            & (self.max_flip_angle >= self.cfg.rewards.flip_success_angle)
            & (self._time() >= self.cfg.rewards.landing_start)
            & (-self.projected_gravity[:, 2] >= self.cfg.rewards.recovery_upright_cos)
            & (self.root_states[:, 2] >= self.cfg.rewards.recovery_min_height)
            & (torch.abs(self.base_ang_vel[:, 1]) <= self.cfg.rewards.recovery_max_pitch_rate)
            & (pose_error <= self.cfg.rewards.recovery_pose_error)
            & (feet_contact_count >= 3)
        )
        self.just_succeeded = (~self.flip_success) & success_now
        self.flip_success |= self.just_succeeded
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.compute_observations()

        self.last_actions_2[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_dof_pos[:] = self.dof_pos
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contact_forces[:] = self.contact_forces
        self.last_torques[:] = self.torques

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        speed_ratio = torch.max(
            torch.abs(self.dof_vel)
            / torch.clamp(
                self.joint_velocity_limits * self.motor_velocity_scales,
                min=1.0,
            ),
            dim=1,
        ).values
        curriculum = self._safety_curriculum_scale()
        normalized_curriculum = np.clip(
            (curriculum - self.cfg.rewards.safety_curriculum_start)
            / (1.0 - self.cfg.rewards.safety_curriculum_start),
            0.0,
            1.0,
        )
        start = self.cfg.control.velocity_termination_start_ratio
        end = self.cfg.control.velocity_termination_ratio
        self.velocity_termination_ratio = start + (end - start) * normalized_curriculum
        self.velocity_termination_buf = speed_ratio > self.velocity_termination_ratio
        self.reset_buf = self.time_out_buf | self.velocity_termination_buf

    def compute_observations(self):
        phase_time = torch.clamp(
            self.episode_length_buf[:, None].float() * self.dt,
            max=self.cfg.rewards.phase_duration,
        )
        phase = torch.pi * phase_time / 2.0
        phase_features = (
            torch.sin(phase), torch.cos(phase),
            torch.sin(phase / 2.0), torch.cos(phase / 2.0),
            torch.sin(phase / 4.0), torch.cos(phase / 4.0),
        )
        sensor_obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
        ), dim=-1)

        actor_sensor = sensor_obs.clone()
        if self.cfg.noise.add_noise:
            actor_sensor[:, 0:3] += (
                self.ang_vel_bias * self.obs_scales.ang_vel
            )
            actor_sensor[:, 3:6] += self.gravity_bias
            actor_sensor[:, 6:18] += (
                self.dof_pos_bias * self.obs_scales.dof_pos
            )
            actor_sensor[:, 18:30] += (
                self.dof_vel_bias * self.obs_scales.dof_vel
            )
            noise = torch.zeros_like(actor_sensor)
            noise[:, 0:3] = self.cfg.noise.noise_scales.ang_vel
            noise[:, 3:6] = self.cfg.noise.noise_scales.gravity
            noise[:, 6:18] = self.cfg.noise.noise_scales.dof_pos
            noise[:, 18:30] = self.cfg.noise.noise_scales.dof_vel
            actor_sensor += (2.0 * torch.rand_like(actor_sensor) - 1.0) * noise

        self.sensor_obs_history = torch.roll(
            self.sensor_obs_history, shifts=1, dims=0
        )
        self.sensor_obs_history[0] = actor_sensor
        fill_ids = self.obs_delay_needs_fill.nonzero(as_tuple=False).flatten()
        if len(fill_ids):
            self.sensor_obs_history[:, fill_ids] = actor_sensor[fill_ids].unsqueeze(0)
            self.obs_delay_needs_fill[fill_ids] = False
        delayed_sensor = self.sensor_obs_history[
            self.obs_delay_steps,
            torch.arange(self.num_envs, device=self.device),
        ]
        self.obs_buf = torch.cat((
            delayed_sensor,
            self.actions,
            self.last_actions,
            *phase_features,
        ), dim=-1)

        state_privileged = torch.cat((
            self.root_states[:, 2:3],
            self.base_lin_vel * self.obs_scales.lin_vel,
            sensor_obs,
            self.actions,
            self.last_actions,
            *phase_features,
        ), dim=-1)

        action_delay = self.env_delay_steps[:, None].float() / max(
            1, self.cfg.control.max_delay_steps
        )
        obs_delay = self.obs_delay_steps[:, None].float() / max(
            1, self.cfg.control.max_observation_delay_steps
        )
        center_scale = max(abs(value) for value in self.cfg.domain_rand.added_center_range)
        motor_offset_scale = max(
            abs(value) for value in self.cfg.domain_rand.added_Motor_OffsetRange
        )
        dynamics_privileged = torch.cat((
            self.torque_scales,
            self.motor_velocity_scales,
            action_delay,
            obs_delay,
            self.limb_mass_scales,
            self.limb_inertia_scales,
            self.contact_friction,
            self.contact_restitution / max(self.cfg.domain_rand.restitution_range[1], 1.0e-6),
            self.contact_offset / 0.01,
            self.base_mass_values
            / float(getattr(self.cfg.domain_rand, "nominal_base_mass", 6.921)),
            self.base_com_values / max(center_scale, 1.0e-6),
            self.p_gains / 40.0,
            self.d_gains,
            self.motor_offsets / max(motor_offset_scale, 1.0e-6),
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((
            state_privileged,
            dynamics_privileged,
        ), dim=-1)
        if self.obs_buf.shape[1] != self.cfg.env.num_observations:
            raise RuntimeError(f"Actor observation shape is {self.obs_buf.shape}")
        if self.privileged_obs_buf.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(
                f"Critic observation shape is {self.privileged_obs_buf.shape}"
            )

        self.obs_dict["obs"] = self.obs_buf
        self.obs_dict["privileged_info"] = self.privileged_obs_buf
        self.obs_dict["priv_info"] = self.priv_info_buf
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.flatten(1)

    def _safety_curriculum_scale(self):
        warmup = self.cfg.rewards.safety_curriculum_warmup_steps
        ramp = max(1, self.cfg.rewards.safety_curriculum_ramp_steps)
        progress = min(max((self.common_step_counter - warmup) / ramp, 0.0), 1.0)
        start = self.cfg.rewards.safety_curriculum_start
        return start + (1.0 - start) * progress

    def _time(self):
        return self.episode_length_buf.float() * self.dt

    def _reward_ang_vel_y(self):
        value = torch.clamp(-self.base_ang_vel[:, 1],
                            -self.cfg.rewards.max_pitch_rate,
                            self.cfg.rewards.max_pitch_rate)
        active = (self._time() > self.cfg.rewards.takeoff_start) & \
                 (self._time() < self.cfg.rewards.rotation_end)
        return value * active

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        value = torch.clamp(self.root_states[:, 9], max=self.cfg.rewards.max_upward_velocity)
        active = (self._time() > self.cfg.rewards.takeoff_start) & \
                 (self._time() < self.cfg.rewards.takeoff_end)
        return value * active

    def _reward_orientation_control(self):
        phase = torch.clamp(self._time() - self.cfg.rewards.takeoff_start, 0.0, 0.5)
        desired_angle = -4.0 * torch.pi * phase
        axis = torch.zeros((self.num_envs, 3), device=self.device)
        axis[:, 1] = 1.0
        desired_quat = quat_from_angle_axis(desired_angle, axis)
        desired_gravity = quat_rotate_inverse(desired_quat, self.gravity_vec)
        return torch.square(self.projected_gravity - desired_gravity).sum(dim=1)

    def _reward_feet_height_before_backflip(self):
        height = torch.clamp(self.feet_pos[:, :, 2] - 0.02, min=0.0)
        return height.sum(dim=1) * (self._time() < self.cfg.rewards.takeoff_start)

    def _reward_height_control(self):
        value = torch.square(self.cfg.rewards.target_height - self.root_states[:, 2])
        active = (self._time() < 0.4) | (self._time() > self.cfg.rewards.landing_start)
        return value * active

    def _reward_default_pose(self):
        """Return to the nominal standing joint pose before takeoff and after landing."""
        error = torch.square(self.dof_pos - self.default_dof_pos).sum(dim=1)
        active = (
            (self._time() < self.cfg.rewards.takeoff_start)
            | (self._time() > self.cfg.rewards.landing_start)
        )
        return error * active

    def _reward_head_clearance(self):
        """Lightly penalize the head approaching the floor before impact."""
        min_head_z = torch.min(
            self.rigid_body_state[:, self.head_indices, 2], dim=1
        ).values
        shortfall = torch.clamp(
            self.cfg.rewards.min_head_center_height - min_head_z, min=0.0
        )
        return self._safety_curriculum_scale() * torch.square(
            shortfall / self.cfg.rewards.min_head_center_height
        )

    def _reward_head_contact(self):
        """Penalize measured head impact force with a curriculum weight."""
        forces = torch.norm(
            self.contact_forces[:, self.head_indices, :], dim=-1
        )
        normalized = torch.clamp(
            torch.max(forces, dim=1).values
            / self.cfg.rewards.head_contact_force,
            min=0.0,
            max=self.cfg.rewards.max_body_contact_penalty,
        )
        return self._safety_curriculum_scale() * normalized

    def _reward_landing_impact(self):
        """Penalize excessive foot impact only during landing/recovery."""
        foot_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        peak_force = torch.max(foot_forces, dim=1).values
        excess = torch.clamp(
            (
                peak_force - self.cfg.rewards.landing_force_threshold
            ) / self.cfg.rewards.landing_force_threshold,
            min=0.0,
            max=self.cfg.rewards.max_landing_impact_penalty,
        )
        active = self._time() > self.cfg.rewards.landing_start
        return (
            self._safety_curriculum_scale()
            * torch.square(excess)
            * active
        )

    def _reward_actions_symmetry(self):
        i = self._dof_index
        pairs = (
            ("FR_hip_joint", "FL_hip_joint", 1.0),
            ("FR_thigh_joint", "FL_thigh_joint", -1.0),
            ("FR_calf_joint", "FL_calf_joint", -1.0),
            ("RR_hip_joint", "RL_hip_joint", 1.0),
            ("RR_thigh_joint", "RL_thigh_joint", -1.0),
            ("RR_calf_joint", "RL_calf_joint", -1.0),
        )
        value = torch.zeros(self.num_envs, device=self.device)
        for right, left, sign in pairs:
            value += torch.square(self.actions[:, i[right]] + sign * self.actions[:, i[left]])
        return value

    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_distance(self):
        relative = self.feet_pos - self.root_states[:, None, 0:3]
        body_feet = torch.empty_like(relative)
        for foot in range(relative.shape[1]):
            body_feet[:, foot] = quat_rotate_inverse(self.base_quat, relative[:, foot])
        return torch.square(body_feet[:, :, 1]).sum(dim=1)

    def _reward_action_rate(self):
        return torch.square(self.actions - self.last_actions).sum(dim=1)

    def _reward_action_jerk(self):
        """Penalize one-step action direction changes that real servos cannot follow."""
        second_difference = (
            self.actions - 2.0 * self.last_actions + self.last_actions_2
        )
        return (
            self._safety_curriculum_scale()
            * torch.square(second_difference).sum(dim=1)
        )

    def _reward_dof_vel_limits(self):
        """Penalize overspeed by squared relative error, without the base cap."""
        effective_limit = (
            self.cfg.control.soft_velocity_limit * self.motor_velocity_scales
        )
        relative_excess = torch.clamp(
            torch.abs(self.dof_vel) / effective_limit - 1.0,
            min=0.0,
            max=4.0,
        )
        return (
            self._safety_curriculum_scale()
            * torch.square(relative_excess).sum(dim=1)
        )

    def _reward_rotation_progress(self):
        """Reward only previously unseen backward rotation."""
        return self.rotation_progress_step

    def _reward_flip_completion(self):
        """One-shot reward for reaching nearly one complete backflip."""
        return self.just_completed.float()

    def _reward_flip_success(self):
        """One-shot reward for completing, landing and regaining the default pose."""
        return self.just_succeeded.float()

    def _recovery_active(self):
        return self.flip_completed & (self._time() >= self.cfg.rewards.landing_start)

    def _reward_recovery_upright(self):
        upright_error = torch.square(self.projected_gravity[:, :2]).sum(dim=1)
        return torch.exp(-4.0 * upright_error) * self._recovery_active()

    def _reward_recovery_height(self):
        error = torch.square(
            (self.root_states[:, 2] - self.cfg.rewards.target_height) / 0.10
        )
        return torch.exp(-error) * self._recovery_active()

    def _reward_recovery_default_pose(self):
        mean_error = torch.square(
            self.dof_pos - self.default_dof_pos
        ).mean(dim=1)
        return torch.exp(-mean_error / 0.09) * self._recovery_active()

    def _reward_recovery_still(self):
        motion = (
            torch.square(self.base_ang_vel).sum(dim=1)
            + 0.25 * torch.square(self.base_lin_vel).sum(dim=1)
        )
        return torch.exp(-0.5 * motion) * self._recovery_active()

    def _reward_recovery_feet_contact(self):
        contact_fraction = torch.mean(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                > self.cfg.rewards.recovery_contact_force
            ).float(),
            dim=1,
        )
        return contact_fraction * self._recovery_active()

    def _reward_rear_leg_action_rate(self):
        """Suppress high-frequency rear-leg commands after takeoff begins."""
        delta = self.actions[:, self.rear_dof_indices] - self.last_actions[
            :, self.rear_dof_indices
        ]
        active = self._time() > self.cfg.rewards.takeoff_start
        return (
            self._safety_curriculum_scale()
            * torch.square(delta).sum(dim=1)
            * active
        )

    def _reward_rear_leg_symmetry(self):
        """Keep the two rear legs coordinated without prescribing a tuck pose."""
        i = self._dof_index
        hip = self.dof_pos[:, i["RR_hip_joint"]] + self.dof_pos[
            :, i["RL_hip_joint"]
        ]
        thigh = self.dof_pos[:, i["RR_thigh_joint"]] - self.dof_pos[
            :, i["RL_thigh_joint"]
        ]
        calf = self.dof_pos[:, i["RR_calf_joint"]] - self.dof_pos[
            :, i["RL_calf_joint"]
        ]
        active = self._time() > self.cfg.rewards.takeoff_start
        return (
            self._safety_curriculum_scale()
            * (torch.square(hip) + torch.square(thigh) + torch.square(calf))
            * active
        )

    def _reward_recovery_dof_velocity(self):
        """Damp leg motion after the nominal landing time."""
        # Normalize by 10 rad/s and cap each joint so an impact cannot dominate
        # the whole PPO batch. This is a motion-quality term, not a hard limit.
        normalized = torch.clamp(torch.abs(self.dof_vel) / 10.0, max=2.0)
        active = self._time() > self.cfg.rewards.landing_start
        return (
            self._safety_curriculum_scale()
            * torch.square(normalized).sum(dim=1)
            * active
        )

    def _reward_undesired_body_contact(self):
        """Penalize ground impacts by the head, trunk, hips or leg links."""
        forces = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
        )
        normalized = torch.clamp(
            forces / self.cfg.rewards.body_contact_force, min=0.0,
            max=self.cfg.rewards.max_body_contact_penalty,
        )
        return (
            self._safety_curriculum_scale()
            * torch.max(normalized, dim=1).values
        )

    def _reward_dof_pos_limits(self):
        return (
            self._safety_curriculum_scale()
            * super()._reward_dof_pos_limits()
        )

    def _reward_joint_target_limits(self):
        """Penalize actor targets that rely on clipping at the safe limits."""
        below = torch.clamp(
            self.dof_pos_limits[:, 0] - self.raw_joint_pos_target, min=0.0
        )
        above = torch.clamp(
            self.raw_joint_pos_target - self.dof_pos_limits[:, 1], min=0.0
        )
        return (
            self._safety_curriculum_scale()
            * torch.square(below + above).sum(dim=1)
        )
