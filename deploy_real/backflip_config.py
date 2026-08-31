"""Validated configuration for the Go2 backflip real-robot controller."""

import os

import numpy as np
import yaml


class BackflipConfig:
    def __init__(self, file_path, actor_override=None):
        self.file_path = os.path.abspath(os.path.expanduser(file_path))
        with open(self.file_path, "r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        if not isinstance(cfg, dict):
            raise ValueError("部署配置必须是 YAML 字典")

        self.lowcmd_topic = str(cfg.get("lowcmd_topic", "rt/lowcmd"))
        self.lowstate_topic = str(cfg.get("lowstate_topic", "rt/lowstate"))
        self.actor_path = self._resolve_path(actor_override or cfg["actor_path"])

        self.leg_joint2motor_idx = np.asarray(
            cfg["leg_joint2motor_idx"], dtype=np.int64
        )
        self.default_angles = self._array(cfg, "default_angles")
        self.kps = self._array(cfg, "kps")
        self.kds = self._array(cfg, "kds")
        self.torque_limits = self._array(cfg, "torque_limits")
        self.brake_torque_limits = self._array(cfg, "brake_torque_limits")
        self.motor_velocity_limits = self._array(cfg, "motor_velocity_limits")
        self.motor_velocity_x1 = float(cfg.get("motor_velocity_x1", 13.5))
        self.torque_limit_enabled = bool(cfg.get("torque_limit_enabled", True))

        self.num_actions = int(cfg.get("num_actions", 12))
        self.num_obs = int(cfg.get("num_obs", 60))
        self.ang_vel_scale = float(cfg.get("ang_vel_scale", 0.25))
        self.dof_pos_scale = float(cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(cfg.get("dof_vel_scale", 0.05))
        self.action_scale = float(cfg.get("action_scale", 0.5))
        self.target_velocity_fraction = float(
            cfg.get("target_velocity_fraction", 0.8)
        )
        self.clip_actions = float(cfg.get("clip_actions", 100.0))
        self.phase_duration = float(cfg.get("phase_duration", 2.0))
        self.control_dt = float(cfg.get("control_dt", 0.02))
        self.lowcmd_dt = float(cfg.get("lowcmd_dt", 0.002))
        self.action_latency_steps = int(cfg.get("action_latency_steps", 1))
        self.diagnostic_interval_steps = int(
            cfg.get("diagnostic_interval_steps", 5)
        )
        self.diagnostic_duration_s = float(
            cfg.get("diagnostic_duration_s", 5.0)
        )

        self.startup_hold_s = float(cfg.get("startup_hold_s", 2.0))
        self.startup_ramp_s = float(cfg.get("startup_ramp_s", 8.0))
        self.startup_stand_s = float(cfg.get("startup_stand_s", 2.0))
        self.blend_in_s = float(cfg.get("blend_in_s", 1.0))
        self.weak_motor = list(cfg.get("weak_motor", []))
        self.damping_kd = float(cfg.get("damping_kd", 8.0))

        self.state_timeout_s = float(cfg.get("state_timeout_s", 0.10))
        self.trigger_max_tilt_rad = float(cfg.get("trigger_max_tilt_rad", 0.25))
        self.trigger_max_joint_speed = float(
            cfg.get("trigger_max_joint_speed", 2.0)
        )
        self.max_joint_speed = float(cfg.get("max_joint_speed", 40.0))
        self.max_abs_action = float(cfg.get("max_abs_action", 10.0))
        self.max_consecutive_late_steps = int(
            cfg.get("max_consecutive_late_steps", 5)
        )
        self.min_battery_voltage = float(
            cfg.get("min_battery_voltage", 24.0)
        )
        self.low_voltage_hold_s = float(
            cfg.get("low_voltage_hold_s", 0.25)
        )
        self.max_motor_lost_increment = int(
            cfg.get("max_motor_lost_increment", 10)
        )
        self.max_zero_temperature_motors = int(
            cfg.get("max_zero_temperature_motors", 3)
        )
        self.require_foot_contact = bool(cfg.get("require_foot_contact", True))
        self.foot_contact_fraction = float(cfg.get("foot_contact_fraction", 0.5))
        self.foot_baseline_min = float(cfg.get("foot_baseline_min", 20.0))

        self._validate()

    def _resolve_path(self, path):
        path = os.path.expandvars(os.path.expanduser(str(path)))
        if os.path.isabs(path):
            return os.path.abspath(path)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.abspath(os.path.join(repo_root, path))

    @staticmethod
    def _array(cfg, key):
        return np.asarray(cfg[key], dtype=np.float32)

    def _validate(self):
        if self.num_actions != 12 or self.num_obs != 60:
            raise ValueError("当前后空翻策略必须是 num_actions=12、num_obs=60")
        if abs(self.control_dt - 0.02) > 1.0e-9:
            raise ValueError("control_dt 必须是训练使用的 0.02 秒（50 Hz）")
        if not 0.001 <= self.lowcmd_dt <= 0.01:
            raise ValueError("lowcmd_dt 必须在宇树建议的 [0.001, 0.01] 秒范围内")
        if self.diagnostic_interval_steps < 1:
            raise ValueError("diagnostic_interval_steps 必须至少为 1")
        if self.diagnostic_duration_s < self.phase_duration:
            raise ValueError("diagnostic_duration_s 不能短于 phase_duration")
        if abs(self.phase_duration - 2.0) > 1.0e-9:
            raise ValueError("phase_duration 必须是训练使用的 2.0 秒")
        if self.action_latency_steps != 1:
            raise ValueError("当前策略训练使用固定一拍延迟，action_latency_steps 必须为 1")
        if sorted(self.leg_joint2motor_idx.tolist()) != list(range(12)):
            raise ValueError("leg_joint2motor_idx 必须恰好包含 0..11")
        for name in (
            "default_angles", "kps", "kds", "torque_limits",
            "brake_torque_limits",
            "motor_velocity_limits",
        ):
            value = getattr(self, name)
            if value.shape != (12,):
                raise ValueError(f"{name} 必须包含 12 个值，实际为 {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} 包含 NaN/Inf")
        if np.any(self.kps < 0.0) or np.any(self.kds < 0.0):
            raise ValueError("Kp/Kd 不能为负数")
        if np.any(self.motor_velocity_limits <= 0.0):
            raise ValueError("motor_velocity_limits 必须为正数")
        if np.any(self.torque_limits <= 0.0) or np.any(
            self.brake_torque_limits <= 0.0
        ):
            raise ValueError("驱动/制动扭矩限制必须为正数")
        if not 0.0 < self.motor_velocity_x1 < float(
            np.min(self.motor_velocity_limits)
        ):
            raise ValueError("motor_velocity_x1 必须在 (0, motor_velocity_limits) 内")
        if self.min_battery_voltage <= 0.0:
            raise ValueError("min_battery_voltage 必须为正数")
        if self.low_voltage_hold_s < 0.0:
            raise ValueError("low_voltage_hold_s 不能为负数")
        if self.max_motor_lost_increment < 0:
            raise ValueError("max_motor_lost_increment 不能为负数")
        if not 0.0 < self.target_velocity_fraction <= 1.0:
            raise ValueError("target_velocity_fraction 必须在 (0, 1] 内")
        if self.state_timeout_s <= self.control_dt:
            raise ValueError("state_timeout_s 必须大于一个控制周期")
