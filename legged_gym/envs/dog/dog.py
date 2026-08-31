import torch

from legged_gym.envs.go2_backflip.go2_backflip import Go2Backflip


class DogBackflip(Go2Backflip):
    """Backflip task adapted to the dog URDF's mirrored joint coordinates."""

    def _head_body_names(self):
        return [name for name in self.body_names if name == "base"]

    def _init_custom_buffers__(self):
        # Keep the legacy base hook name (including its trailing underscores).
        super()._init_custom_buffers__()
        kwargs = {"device": self.device, "dtype": torch.float}
        control = self.cfg.control
        self.dog_motor_torque_limits = torch.tensor(
            control.motor_torque_limits, **kwargs
        ).unsqueeze(0)
        self.dog_joint_velocity_limits = torch.tensor(
            control.joint_velocity_limits, **kwargs
        ).unsqueeze(0)
        self.dog_target_velocity_limits = torch.tensor(
            control.target_velocity_limits, **kwargs
        ).unsqueeze(0)
        self.dog_motor_velocity_x1 = torch.tensor(
            control.motor_velocity_x1, **kwargs
        ).unsqueeze(0)
        self.dog_soft_velocity_limits = torch.tensor(
            control.soft_velocity_limits, **kwargs
        ).unsqueeze(0)

    def _compute_torques(self, actions):
        """PD control with randomized delay and a URDF-derived torque envelope."""
        delayed_actions = self.action_buffer[
            self.env_delay_steps,
            torch.arange(self.num_envs, device=self.device),
        ]
        max_action_delta = (
            self.dog_target_velocity_limits
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
        self.raw_joint_pos_target = (
            delayed_actions * self.cfg.control.action_scale
            + self.default_dof_pos
            + self.motor_offsets
        )
        self.joint_pos_target = (
            self.slew_limited_actions * self.cfg.control.action_scale
            + self.default_dof_pos
            + self.motor_offsets
        )
        torques = (
            self.p_gains * (self.joint_pos_target - self.dof_pos)
            - self.d_gains * self.dof_vel
        )

        velocity_scale = self.motor_velocity_scales
        x1 = self.dog_motor_velocity_x1 * velocity_scale
        x2 = self.dog_joint_velocity_limits * velocity_scale
        speed = torch.abs(self.dof_vel)
        speed_fraction = torch.where(
            speed < x1,
            torch.ones_like(speed),
            torch.clamp(
                (x2 - speed) / torch.clamp(x2 - x1, min=1.0e-6),
                min=0.0,
                max=1.0,
            ),
        )
        torque_limit = (
            self.dog_motor_torque_limits
            * self.torque_scales
            * speed_fraction
        )
        return torch.clamp(torques, min=-torque_limit, max=torque_limit)

    def _reward_actions_symmetry(self):
        # A symmetric physical pose has opposite left/right URDF coordinates.
        front = sum(
            torch.square(self.actions[:, i] + self.actions[:, i + 3])
            for i in range(3)
        )
        rear = sum(
            torch.square(self.actions[:, i] + self.actions[:, i + 3])
            for i in range(6, 9)
        )
        return front + rear

    def _reward_rear_leg_symmetry(self):
        return sum(
            torch.square(self.actions[:, i] + self.actions[:, i + 3])
            for i in range(6, 9)
        )

    def _reward_dof_vel_limits(self):
        effective_limits = (
            self.dog_soft_velocity_limits * self.motor_velocity_scales
        )
        excess = torch.clamp(
            torch.abs(self.dof_vel) - effective_limits, min=0.0
        )
        return torch.sum(excess, dim=1)
