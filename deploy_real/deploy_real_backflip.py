#!/usr/bin/env python3
"""Deploy the 60-dimensional phase-conditioned Go2 backflip actor.

Remote controls:
  START  arm policy control after the startup stance
  A      trigger one two-second backflip phase
  SELECT emergency damping stop

A trigger resets only the policy phase. The actor continuously controls takeoff,
landing, recovery and the waiting stance; robot state is never reset between flips.
"""

import argparse
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
LOCAL_SDK = os.path.join(REPO_ROOT, "unitree_sdk2_python")
if os.path.isdir(LOCAL_SDK):
    sys.path.insert(0, LOCAL_SDK)

from backflip_config import BackflipConfig
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation

POLICY_JOINT_NAMES = (
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
)

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


class SafetyStop(RuntimeError):
    pass


class BackflipPolicy:
    def __init__(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ONNX 不存在: {path}\n"
                "请先用 mujoco/script/pt2onnx.py 导出 actor。"
            )
        self.session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise ValueError("后空翻 actor 必须只有一个输入和至少一个输出")
        if inputs[0].shape[-1] != 60:
            raise ValueError(f"ONNX 输入应为 60 维，实际为 {inputs[0].shape}")
        if outputs[0].shape[-1] != 12:
            raise ValueError(f"ONNX 输出应为 12 维，实际为 {outputs[0].shape}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.path = path

    def act(self, observation):
        action = self.session.run(
            [self.output_name],
            {self.input_name: observation[None, :].astype(np.float32)},
        )[0][0]
        return np.asarray(action, dtype=np.float32)


def load_unitree_sdk():
    try:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__LowCmd_,
            unitree_go_msg_dds__LowState_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
        from unitree_sdk2py.utils.crc import CRC
        from common.command_helper import init_cmd_go
    except ImportError as exc:
        raise ImportError(
            "缺少 unitree_sdk2py。请按 deploy_real/README.md 安装 SDK。"
        ) from exc
    return SimpleNamespace(
        ChannelFactoryInitialize=ChannelFactoryInitialize,
        ChannelPublisher=ChannelPublisher,
        ChannelSubscriber=ChannelSubscriber,
        LowCmdDefault=unitree_go_msg_dds__LowCmd_,
        LowStateDefault=unitree_go_msg_dds__LowState_,
        LowCmdGo=LowCmdGo,
        LowStateGo=LowStateGo,
        CRC=CRC,
        init_cmd_go=init_cmd_go,
    )


class BackflipController:
    def __init__(self, config, policy, sdk, verbose=False):
        self.config = config
        self.policy = policy
        self.sdk = sdk
        self.verbose = verbose
        self.remote = RemoteController()

        self.low_cmd = sdk.LowCmdDefault()
        self.low_state = sdk.LowStateDefault()
        self.last_state_time = 0.0
        self.received_state = False
        self.crc = sdk.CRC()
        self.command_lock = threading.Lock()
        self.lowcmd_stop = threading.Event()
        self.lowcmd_thread = None
        self.lowcmd_error = None
        self.lowcmd_publish_count = 0
        self.lowcmd_late_count = 0
        self.lowcmd_worst_lateness = 0.0

        self.publisher = sdk.ChannelPublisher(
            config.lowcmd_topic, sdk.LowCmdGo
        )
        self.publisher.Init()
        self.subscriber = sdk.ChannelSubscriber(
            config.lowstate_topic, sdk.LowStateGo
        )
        self.subscriber.Init(self._low_state_handler, 10)
        sdk.init_cmd_go(self.low_cmd, weak_motor=config.weak_motor)

        self.q = np.zeros(12, dtype=np.float32)
        self.dq = np.zeros(12, dtype=np.float32)
        self.tau_est = np.zeros(12, dtype=np.float32)
        self.motor_temperature = np.zeros(12, dtype=np.float32)
        self.motor_lost = np.zeros(12, dtype=np.uint32)
        self.last_gyro = np.zeros(3, dtype=np.float32)
        self.last_target = config.default_angles.copy()
        self.desired_kps = config.kps.copy()
        self.desired_kds = config.kds.copy()
        self.torque_control_active = False
        self.direct_torque_active = False
        self.last_target_update_time = time.monotonic()
        self.limited_target = config.default_angles.copy()
        self.raw_pd_torque = np.zeros(12, dtype=np.float32)
        self.limited_torque = np.zeros(12, dtype=np.float32)
        self.dynamic_torque_limit = config.torque_limits.copy()
        self.motor_lost_baseline = None
        self.current_action = np.zeros(12, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.slew_limited_action = np.zeros(12, dtype=np.float32)
        self.phase_step = 0
        self.phase_steps = int(round(config.phase_duration / config.control_dt))
        self.backflip_active = False
        self.policy_enabled_at = None
        self.enable_pose = None
        self.foot_force_baseline = 0.0
        self.counter = 0
        self.late_steps = 0
        self.consecutive_late_steps = 0
        self.max_period = 0.0
        self.flip_diagnostics = None
        self.inference_time_ms = 0.0
        self.writer_min_voltage = float("inf")
        self.writer_max_abs_current = 0.0
        self.low_voltage_since = None
        self.low_voltage_warning_active = False
        self.writer_q = np.zeros(12, dtype=np.float32)
        self.writer_dq = np.zeros(12, dtype=np.float32)
        self.writer_tau_est = np.zeros(12, dtype=np.float32)
        self.writer_sample_time = 0.0

    def _publish_lowcmd_once(self):
        snapshot = self._writer_state_snapshot()
        with self.command_lock:
            if snapshot is not None:
                q, dq, tau_est, temperature, lost, voltage, current = snapshot
                self.writer_q[:] = q
                self.writer_dq[:] = dq
                self.writer_tau_est[:] = tau_est
                self.writer_sample_time = time.monotonic()
                if self.flip_diagnostics is not None:
                    if np.isfinite(voltage):
                        self.writer_min_voltage = min(
                            self.writer_min_voltage, voltage
                        )
                    if np.isfinite(current):
                        self.writer_max_abs_current = max(
                            self.writer_max_abs_current, abs(current)
                        )
                health_error = None
                if self.direct_torque_active:
                    state_age = time.monotonic() - self.last_state_time
                    target_age = time.monotonic() - self.last_target_update_time
                    if state_age > self.config.state_timeout_s:
                        health_error = (
                            f"LowState 超时 {state_age * 1000.0:.0f} ms"
                        )
                    elif target_age > self.config.state_timeout_s:
                        health_error = (
                            f"策略力矩目标超时 {target_age * 1000.0:.0f} ms"
                        )
                if health_error is None:
                    health_error = self._motor_health_error(
                        temperature, lost, voltage
                    )
                if health_error is not None and self.lowcmd_error is None:
                    self.lowcmd_error = health_error
                    self._set_damping_command_locked()
                    print(
                        f"[WATCHDOG] {health_error}; "
                        "500-Hz publisher switched to damping",
                        flush=True,
                    )
                elif self.torque_control_active:
                    (
                        limited_target,
                        raw_torque,
                        limited_torque,
                        dynamic_limit,
                    ) = self._limit_pd_command(
                        q, dq, self.last_target,
                        self.desired_kps, self.desired_kds,
                    )
                    self.limited_target[:] = limited_target
                    self.raw_pd_torque[:] = raw_torque
                    self.limited_torque[:] = limited_torque
                    self.dynamic_torque_limit[:] = dynamic_limit
                    for i, motor_id in enumerate(
                        self.config.leg_joint2motor_idx
                    ):
                        motor = self.low_cmd.motor_cmd[int(motor_id)]
                        if self.direct_torque_active:
                            # Match training: send the final clipped effort
                            # without a second motor-side PD calculation.
                            motor.q = POS_STOP_F
                            motor.dq = VEL_STOP_F
                            motor.kp = 0.0
                            motor.kd = 0.0
                            motor.tau = float(limited_torque[i])
                        else:
                            # Startup and arming remain position controlled.
                            motor.q = float(limited_target[i])
                            motor.dq = 0.0
                            motor.kp = float(self.desired_kps[i])
                            motor.kd = float(self.desired_kds[i])
                            motor.tau = 0.0
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.publisher.Write(self.low_cmd)

    def _writer_state_snapshot(self):
        if not self.received_state:
            return None
        message = self.low_state
        q = np.empty(12, dtype=np.float32)
        dq = np.empty(12, dtype=np.float32)
        tau_est = np.empty(12, dtype=np.float32)
        temperature = np.empty(12, dtype=np.float32)
        lost = np.empty(12, dtype=np.int64)
        for i, motor_id in enumerate(self.config.leg_joint2motor_idx):
            state = message.motor_state[int(motor_id)]
            q[i] = state.q
            dq[i] = state.dq
            tau_est[i] = state.tau_est
            temperature[i] = state.temperature
            lost[i] = state.lost
        voltage = float(getattr(message, "power_v", float("nan")))
        current = float(getattr(message, "power_a", float("nan")))
        return q, dq, tau_est, temperature, lost, voltage, current

    def _motor_health_error(self, temperature, lost, voltage):
        if self.motor_lost_baseline is None:
            self.motor_lost_baseline = lost.copy()
        lost_increment = lost - self.motor_lost_baseline
        max_lost = int(np.max(lost_increment))
        if max_lost > self.config.max_motor_lost_increment:
            joint = POLICY_JOINT_NAMES[int(np.argmax(lost_increment))]
            return (
                f"电机状态丢失: {joint} lost增加 {max_lost}，"
                f"阈值 {self.config.max_motor_lost_increment}"
            )
        zero_temperature_count = int(np.sum(temperature <= 0.0))
        if zero_temperature_count > self.config.max_zero_temperature_motors:
            return f"电机状态失效: {zero_temperature_count}/12 个温度为0"
        if not self.torque_control_active or not np.isfinite(voltage):
            self.low_voltage_since = None
            self.low_voltage_warning_active = False
            return None

        now = time.monotonic()
        if voltage < self.config.min_battery_voltage:
            if self.low_voltage_since is None:
                self.low_voltage_since = now
            low_duration = now - self.low_voltage_since
            if not self.low_voltage_warning_active:
                print(
                    f"[VOLTAGE] transient dip {voltage:.2f} V < "
                    f"{self.config.min_battery_voltage:.2f} V; "
                    f"waiting {self.config.low_voltage_hold_s:.2f}s before stop",
                    flush=True,
                )
                self.low_voltage_warning_active = True
            if low_duration >= self.config.low_voltage_hold_s:
                return (
                    f"电池电压持续过低 {low_duration:.3f}s: "
                    f"{voltage:.2f} V < "
                    f"{self.config.min_battery_voltage:.2f} V"
                )
        elif self.low_voltage_since is not None:
            low_duration = now - self.low_voltage_since
            print(
                f"[VOLTAGE] recovered to {voltage:.2f} V after "
                f"{low_duration:.3f}s below threshold; continuing",
                flush=True,
            )
            self.low_voltage_since = None
            self.low_voltage_warning_active = False
        return None

    def _limit_pd_command(self, q, dq, target, kps, kds):
        raw_torque = kps * (target - q) - kds * dq
        if not self.config.torque_limit_enabled:
            return target.copy(), raw_torque, raw_torque.copy(), np.full(
                12, np.inf, dtype=np.float32
            )
        speed = np.abs(dq)
        x1 = self.config.motor_velocity_x1
        x2 = self.config.motor_velocity_limits
        speed_fraction = np.where(
            speed < x1,
            1.0,
            np.clip((x2 - speed) / (x2 - x1), 0.0, 1.0),
        )
        same_direction = dq * raw_torque > 0.0
        peak_torque = np.where(
            same_direction,
            self.config.torque_limits,
            self.config.brake_torque_limits,
        )
        dynamic_limit = peak_torque * speed_fraction
        limited_torque = np.clip(
            raw_torque, -dynamic_limit, dynamic_limit
        )
        limited_target = q.copy()
        active_kp = kps > 1.0e-6
        limited_target[active_kp] = (
            q[active_kp]
            + (
                limited_torque[active_kp]
                + kds[active_kp] * dq[active_kp]
            ) / kps[active_kp]
        )
        return (
            limited_target.astype(np.float32),
            raw_torque.astype(np.float32),
            limited_torque.astype(np.float32),
            dynamic_limit.astype(np.float32),
        )

    def _set_damping_command_locked(self):
        self.torque_control_active = False
        self.direct_torque_active = False
        for motor in self.low_cmd.motor_cmd:
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = self.config.damping_kd
            motor.tau = 0.0

    def _lowcmd_writer_loop(self):
        next_deadline = time.perf_counter()
        try:
            while not self.lowcmd_stop.is_set():
                self._publish_lowcmd_once()
                self.lowcmd_publish_count += 1
                next_deadline += self.config.lowcmd_dt
                remaining = next_deadline - time.perf_counter()
                if remaining > 0.0:
                    self.lowcmd_stop.wait(remaining)
                else:
                    lateness = -remaining
                    self.lowcmd_late_count += 1
                    self.lowcmd_worst_lateness = max(
                        self.lowcmd_worst_lateness, lateness
                    )
                    next_deadline = time.perf_counter()
        except Exception as exc:
            self.lowcmd_error = exc
            self.lowcmd_stop.set()

    def start_lowcmd_writer(self):
        if self.lowcmd_thread is not None:
            return
        self.lowcmd_stop.clear()
        self.lowcmd_thread = threading.Thread(
            target=self._lowcmd_writer_loop,
            name="go2-lowcmd-500hz",
            daemon=True,
        )
        self.lowcmd_thread.start()
        print(
            f"[LOWCMD] publisher started at "
            f"{1.0 / self.config.lowcmd_dt:.0f} Hz; actor remains at "
            f"{1.0 / self.config.control_dt:.0f} Hz"
        )
        print(
            "[ACTUATOR] Go2HV torque limiting: "
            f"drive={self.config.torque_limits[0]:.1f} Nm, "
            f"brake={self.config.brake_torque_limits[0]:.1f} Nm, "
            f"X1={self.config.motor_velocity_x1:.1f} rad/s, "
            f"X2={self.config.motor_velocity_limits[0]:.1f} rad/s"
        )
        print(
            "[ACTUATOR] policy uses direct MotorCmd.tau at 500 Hz; "
            "startup remains motor-side position PD"
        )

    def stop_lowcmd_writer(self):
        self.lowcmd_stop.set()
        if self.lowcmd_thread is not None:
            self.lowcmd_thread.join(timeout=1.0)
            self.lowcmd_thread = None

    def _low_state_handler(self, msg):
        self.low_state = msg
        self.last_state_time = time.monotonic()
        self.received_state = True
        self.remote.set(self.low_state.wireless_remote)

    def send_cmd(self):
        # Before the recurrent writer starts, publish synchronously. Once it is
        # active, callers only update the shared command and the 500-Hz thread
        # republishes the latest complete packet.
        if self.lowcmd_thread is None:
            self._publish_lowcmd_once()

    def _check_operator_and_link(self):
        if self.remote.is_down(KeyMap.select):
            raise SafetyStop("SELECT pressed")
        state_age = time.monotonic() - self.last_state_time
        if state_age > self.config.state_timeout_s:
            raise SafetyStop(f"LowState 超时 {state_age * 1000.0:.0f} ms")

    def wait_for_low_state(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while not self.received_state:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{timeout:.0f}s 内没有收到 {self.config.lowstate_topic}"
                )
            self.send_cmd()
            time.sleep(self.config.control_dt)
        print("[DDS] LowState connected")
        self._read_joint_state()
        gravity, _ = self._imu()
        print(f"[DDS] q={np.round(self.q, 3)}")
        print(f"[DDS] projected_gravity={np.round(gravity, 4)}")

    def _read_joint_state(self):
        idx = self.config.leg_joint2motor_idx
        for i, motor_id in enumerate(idx):
            state = self.low_state.motor_state[int(motor_id)]
            self.q[i] = state.q
            self.dq[i] = state.dq
            self.tau_est[i] = state.tau_est
            self.motor_temperature[i] = state.temperature
            self.motor_lost[i] = state.lost
        if not np.all(np.isfinite(self.q)) or not np.all(np.isfinite(self.dq)):
            raise SafetyStop("关节状态包含 NaN/Inf")

    def _imu(self):
        quat_raw = self.low_state.imu_state.quaternion
        quat = np.asarray(
            [quat_raw[0], quat_raw[1], quat_raw[2], quat_raw[3]],
            dtype=np.float32,
        )
        gyro = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        if not np.all(np.isfinite(quat)) or not np.all(np.isfinite(gyro)):
            raise SafetyStop("IMU 状态包含 NaN/Inf")
        return get_gravity_orientation(quat).astype(np.float32), gyro

    def _tilt(self, projected_gravity):
        # Upright gravity is [0, 0, -1] -> 0 rad; fully inverted -> pi rad.
        # asin(norm(g_xy)) cannot distinguish these two cases.
        return float(
            np.arccos(np.clip(-float(projected_gravity[2]), -1.0, 1.0))
        )

    def _foot_force_sum(self):
        try:
            values = np.asarray(self.low_state.foot_force[:4], dtype=np.float32)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            return 0.0
        return float(np.maximum(values, 0.0).sum())

    def _foot_forces(self):
        try:
            values = np.asarray(self.low_state.foot_force[:4], dtype=np.float32)
        except (AttributeError, TypeError, ValueError):
            return np.zeros(4, dtype=np.float32)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            return np.zeros(4, dtype=np.float32)
        return np.maximum(values, 0.0)

    def _send_position(self, target, kps=None, kds=None):
        kps = self.config.kps if kps is None else kps
        kds = self.config.kds if kds is None else kds
        with self.command_lock:
            self.last_target[:] = target
            self.desired_kps[:] = kps
            self.desired_kds[:] = kds
            self.torque_control_active = True
            self.last_target_update_time = time.monotonic()
        self.send_cmd()

    def hold_current_pose(self):
        self._read_joint_state()
        self._send_position(self.q.copy())

    def startup(self):
        self._read_joint_state()
        initial = self.q.copy()
        print(
            f"[1/3] Hold current pose for {self.config.startup_hold_s:.1f}s"
        )
        for _ in range(int(self.config.startup_hold_s / self.config.control_dt)):
            self._check_operator_and_link()
            self._send_position(initial)
            time.sleep(self.config.control_dt)

        print(
            f"[2/3] Ramp to trained default pose over "
            f"{self.config.startup_ramp_s:.1f}s"
        )
        count = int(self.config.startup_ramp_s / self.config.control_dt)
        for step in range(count):
            self._check_operator_and_link()
            alpha = 0.5 * (1.0 - np.cos(np.pi * step / max(count - 1, 1)))
            target = (
                initial * (1.0 - alpha) + self.config.default_angles * alpha
            )
            self._send_position(target)
            time.sleep(self.config.control_dt)

        print(
            f"[3/3] Hold default pose for {self.config.startup_stand_s:.1f}s"
        )
        foot_samples = []
        count = int(self.config.startup_stand_s / self.config.control_dt)
        for _ in range(count):
            self._check_operator_and_link()
            self._send_position(self.config.default_angles)
            foot_samples.append(self._foot_force_sum())
            time.sleep(self.config.control_dt)
        self._read_joint_state()
        error = float(np.max(np.abs(self.q - self.config.default_angles)))
        if error > 0.25:
            raise SafetyStop(f"默认站姿未收敛，最大关节误差 {error:.3f} rad")
        positive_samples = [value for value in foot_samples if value > 0.0]
        self.foot_force_baseline = (
            float(np.median(positive_samples)) if positive_samples else 0.0
        )
        print(f"[STARTUP] max joint error={error:.3f} rad")
        print(f"[STARTUP] foot-force baseline={self.foot_force_baseline:.1f}")

    def wait_for_arm(self):
        # Drop button edges generated during startup.
        self.remote.is_pressed(KeyMap.A)
        self.remote.is_pressed(KeyMap.start)
        print("[ARM] Press START to enable RL; SELECT aborts to damping")
        while True:
            self._check_operator_and_link()
            if self.remote.is_pressed(KeyMap.start):
                return
            self._send_position(self.config.default_angles)
            time.sleep(self.config.control_dt)

    def enable_policy(self):
        self._read_joint_state()
        self.enable_pose = self.q.copy()
        self.policy_enabled_at = time.monotonic()
        self.current_action.fill(0.0)
        self.last_action.fill(0.0)
        self.slew_limited_action.fill(0.0)
        self.phase_step = 0
        self.backflip_active = False
        with self.command_lock:
            self.direct_torque_active = True
            self.last_target_update_time = time.monotonic()
        print("[POLICY] direct torque enabled at phase 0; waiting for A")

    def _phase_features(self):
        phase_time = self.phase_step * self.config.control_dt
        phase = np.pi * phase_time / 2.0
        return np.asarray(
            [
                np.sin(phase), np.cos(phase),
                np.sin(phase / 2.0), np.cos(phase / 2.0),
                np.sin(phase / 4.0), np.cos(phase / 4.0),
            ],
            dtype=np.float32,
        )

    def build_observation(self):
        self._read_joint_state()
        projected_gravity, gyro = self._imu()
        self.last_gyro[:] = gyro
        observation = np.concatenate(
            (
                gyro * self.config.ang_vel_scale,
                projected_gravity,
                (self.q - self.config.default_angles)
                * self.config.dof_pos_scale,
                self.dq * self.config.dof_vel_scale,
                self.current_action,
                self.last_action,
                self._phase_features(),
            )
        ).astype(np.float32)
        if observation.shape != (60,):
            raise SafetyStop(f"观测维度错误: {observation.shape}")
        if not np.all(np.isfinite(observation)):
            raise SafetyStop("策略观测包含 NaN/Inf")
        return np.clip(observation, -100.0, 100.0), projected_gravity

    def safety_check(self, projected_gravity=None):
        if self.lowcmd_error is not None:
            raise SafetyStop(f"LowCmd 发布线程异常: {self.lowcmd_error}")
        state_age = time.monotonic() - self.last_state_time
        if state_age > self.config.state_timeout_s:
            raise SafetyStop(f"LowState 超时 {state_age * 1000.0:.0f} ms")
        self._read_joint_state()
        if float(np.max(np.abs(self.dq))) > self.config.max_joint_speed:
            raise SafetyStop(
                f"关节速度过大: {float(np.max(np.abs(self.dq))):.1f} rad/s"
            )
    def _trigger_ready(self, projected_gravity):
        if self.backflip_active:
            return False, "previous phase is still active"
        tilt = self._tilt(projected_gravity)
        if tilt > self.config.trigger_max_tilt_rad:
            return False, f"tilt {tilt:.2f} rad"
        max_speed = float(np.max(np.abs(self.dq)))
        if max_speed > self.config.trigger_max_joint_speed:
            return False, f"joint speed {max_speed:.2f} rad/s"
        if self.config.require_foot_contact:
            if self.foot_force_baseline < self.config.foot_baseline_min:
                return False, "foot-force baseline is unavailable"
            foot_sum = self._foot_force_sum()
            threshold = (
                self.config.foot_contact_fraction * self.foot_force_baseline
            )
            if foot_sum < threshold:
                return False, f"foot force {foot_sum:.1f} < {threshold:.1f}"
        return True, "ready"

    def _validate_action(self, action):
        if not np.all(np.isfinite(action)):
            raise SafetyStop("actor 输出包含 NaN/Inf")
        action_peak = float(np.max(np.abs(action)))
        if action_peak > self.config.max_abs_action:
            raise SafetyStop(f"actor 输出异常: max |action|={action_peak:.2f}")

    @staticmethod
    def _peak_update(diagnostics, key, value, joint, elapsed):
        if value > diagnostics[key][0]:
            diagnostics[key] = (float(value), str(joint), float(elapsed))

    def _start_flip_diagnostics(self):
        self.writer_min_voltage = float("inf")
        self.writer_max_abs_current = 0.0
        self.flip_diagnostics = {
            "start_counter": self.counter,
            "samples": 0,
            "max_tilt": (0.0, "base", 0.0),
            "max_pitch_rate": (0.0, "imu_y", 0.0),
            "max_joint_speed": (0.0, "", 0.0),
            "max_joint_error": (0.0, "", 0.0),
            "max_pd_torque": (0.0, "", 0.0),
            "max_torque_ratio": (0.0, "", 0.0),
            "max_tau_command": (0.0, "", 0.0),
            "max_tau_est": (0.0, "", 0.0),
            "max_tau_tracking_error": (0.0, "", 0.0),
            "max_action": (0.0, "", 0.0),
            "cumulative_pitch": 0.0,
            "max_backward_pitch": 0.0,
            "airborne_time": None,
            "landing_time": None,
            "max_inference_ms": 0.0,
            "max_state_age_ms": 0.0,
            "min_voltage": float("inf"),
            "max_abs_current": 0.0,
            "min_foot_sum": float("inf"),
            "max_foot_sum": 0.0,
        }
        print(
            "[DIAG] 5-second capture started; compact rows at 10 Hz, "
            "full vectors at 2 Hz"
        )

    def _update_flip_diagnostics(self, projected_gravity, target, action):
        diagnostics = self.flip_diagnostics
        if diagnostics is None:
            return
        elapsed = (
            self.counter - diagnostics["start_counter"]
        ) * self.config.control_dt
        tilt = self._tilt(projected_gravity)
        joint_error = target - self.q
        pd_torque = self.config.kps * joint_error - self.config.kds * self.dq
        torque_ratio_vector = np.abs(pd_torque) / self.config.torque_limits
        with self.command_lock:
            limited_target = self.limited_target.copy()
            raw_pd_torque = self.raw_pd_torque.copy()
            limited_torque = self.limited_torque.copy()
            dynamic_limit = self.dynamic_torque_limit.copy()
            writer_tau_est = self.writer_tau_est.copy()
            writer_sample_time = self.writer_sample_time
        torque_tracking_error = writer_tau_est - limited_torque
        saturated = (
            np.abs(raw_pd_torque) > dynamic_limit + 1.0e-3
        )
        saturated_count = int(np.sum(saturated))
        foot_forces = self._foot_forces()
        foot_sum = float(foot_forces.sum())
        voltage = float(getattr(self.low_state, "power_v", float("nan")))
        current = float(getattr(self.low_state, "power_a", float("nan")))
        state_age_ms = (time.monotonic() - self.last_state_time) * 1000.0
        diagnostics["cumulative_pitch"] += (
            -float(self.last_gyro[1]) * self.config.control_dt
        )
        diagnostics["max_backward_pitch"] = max(
            diagnostics["max_backward_pitch"],
            diagnostics["cumulative_pitch"],
        )
        contact_threshold = max(
            5.0, self.foot_force_baseline * self.config.foot_contact_fraction
        )
        if (
            diagnostics["airborne_time"] is None
            and foot_sum < contact_threshold
        ):
            diagnostics["airborne_time"] = elapsed
            print(f"[DIAG EVENT] airborne at t={elapsed:.2f}s, foot={foot_sum:.1f}")
        elif (
            diagnostics["airborne_time"] is not None
            and diagnostics["landing_time"] is None
            and elapsed > diagnostics["airborne_time"] + 0.05
            and foot_sum >= contact_threshold
        ):
            diagnostics["landing_time"] = elapsed
            print(f"[DIAG EVENT] foot contact restored at t={elapsed:.2f}s, foot={foot_sum:.1f}")

        def peak_from_vector(key, vector):
            index = int(np.argmax(np.abs(vector)))
            self._peak_update(
                diagnostics, key, abs(float(vector[index])),
                POLICY_JOINT_NAMES[index], elapsed,
            )

        self._peak_update(diagnostics, "max_tilt", tilt, "base", elapsed)
        self._peak_update(
            diagnostics, "max_pitch_rate", abs(float(self.last_gyro[1])),
            "imu_y", elapsed,
        )
        peak_from_vector("max_joint_speed", self.dq)
        peak_from_vector("max_joint_error", joint_error)
        peak_from_vector("max_pd_torque", pd_torque)
        peak_from_vector("max_torque_ratio", torque_ratio_vector)
        peak_from_vector("max_tau_command", limited_torque)
        peak_from_vector("max_tau_est", writer_tau_est)
        peak_from_vector("max_tau_tracking_error", torque_tracking_error)
        peak_from_vector("max_action", action)
        diagnostics["max_inference_ms"] = max(
            diagnostics["max_inference_ms"], self.inference_time_ms
        )
        diagnostics["max_state_age_ms"] = max(
            diagnostics["max_state_age_ms"], state_age_ms
        )
        if np.isfinite(voltage):
            diagnostics["min_voltage"] = min(
                diagnostics["min_voltage"], voltage
            )
        if np.isfinite(current):
            diagnostics["max_abs_current"] = max(
                diagnostics["max_abs_current"], abs(current)
            )
        diagnostics["min_foot_sum"] = min(
            diagnostics["min_foot_sum"], foot_sum
        )
        diagnostics["max_foot_sum"] = max(
            diagnostics["max_foot_sum"], foot_sum
        )
        diagnostics["samples"] += 1

        interval = self.config.diagnostic_interval_steps
        if (self.counter - diagnostics["start_counter"]) % interval == 0:
            dq_index = int(np.argmax(np.abs(self.dq)))
            error_index = int(np.argmax(np.abs(joint_error)))
            pd_index = int(np.argmax(torque_ratio_vector))
            tau_index = int(np.argmax(np.abs(writer_tau_est)))
            writer_age_ms = max(
                0.0, (time.monotonic() - writer_sample_time) * 1000.0
            )
            print(
                f"[DIAG] t={elapsed:4.2f}s tilt={tilt:4.2f} "
                f"gyro={np.round(self.last_gyro, 2)} "
                f"pitch={diagnostics['cumulative_pitch']:+.2f} "
                f"maxdq={POLICY_JOINT_NAMES[dq_index]}:{self.dq[dq_index]:+.2f} "
                f"qerr={POLICY_JOINT_NAMES[error_index]}:{joint_error[error_index]:+.3f} "
                f"PD={POLICY_JOINT_NAMES[pd_index]}:{pd_torque[pd_index]:+.1f}Nm/"
                f"{torque_ratio_vector[pd_index]:.2f}x "
                f"cmd={limited_torque[pd_index]:+.1f}/"
                f"{dynamic_limit[pd_index]:.1f}Nm sat={saturated_count}/12 "
                f"tau_est={POLICY_JOINT_NAMES[tau_index]}:{writer_tau_est[tau_index]:+.1f}Nm "
                f"act={np.max(np.abs(action)):.2f} foot={foot_sum:.0f} "
                f"V/A={voltage:.1f}/{current:+.1f} "
                f"state={state_age_ms:.1f}ms writer={writer_age_ms:.1f}ms "
                f"infer={self.inference_time_ms:.2f}ms",
                flush=True,
            )
        if (self.counter - diagnostics["start_counter"]) % 25 == 0:
            print(
                f"[DIAG-VEC] q={np.round(self.q, 3)}\n"
                f"           dq={np.round(self.dq, 2)}\n"
                f"       target={np.round(target, 3)}\n"
                f"      limited={np.round(limited_target, 3)}\n"
                f"       action={np.round(action, 3)}\n"
                f"       tau_cmd={np.round(limited_torque, 1)}\n"
                f"      tau_est={np.round(writer_tau_est, 1)}\n"
                f"       tau_err={np.round(torque_tracking_error, 1)} "
                f"foot={np.round(foot_forces, 0)} "
                f"temp={self.motor_temperature.astype(int)} lost={self.motor_lost}",
                flush=True,
            )
        if elapsed >= self.config.diagnostic_duration_s:
            self.print_flip_diagnostics_summary()

    def print_flip_diagnostics_summary(self):
        diagnostics = self.flip_diagnostics
        if diagnostics is None:
            return
        print("[DIAG SUMMARY] peak value | source | seconds after A")
        for key in (
            "max_tilt", "max_pitch_rate", "max_joint_speed",
            "max_joint_error", "max_pd_torque", "max_torque_ratio",
            "max_tau_command", "max_tau_est", "max_tau_tracking_error",
            "max_action",
        ):
            value, source, elapsed = diagnostics[key]
            print(f"  {key:20s} {value:8.3f} | {source:9s} | {elapsed:5.2f}s")
        min_voltage = min(
            diagnostics["min_voltage"], self.writer_min_voltage
        )
        voltage_text = f"{min_voltage:.2f}" if np.isfinite(min_voltage) else "n/a"
        print(
            f"  voltage min={voltage_text}V, "
            f"|current| max={max(diagnostics['max_abs_current'], self.writer_max_abs_current):.2f}A, "
            f"foot sum={diagnostics['min_foot_sum']:.1f}--"
            f"{diagnostics['max_foot_sum']:.1f}, "
            f"state age max={diagnostics['max_state_age_ms']:.2f}ms, "
            f"inference max={diagnostics['max_inference_ms']:.2f}ms"
        )
        print(
            f"  pitch integrated={diagnostics['cumulative_pitch']:+.3f} rad, "
            f"max backward={diagnostics['max_backward_pitch']:.3f} rad, "
            f"airborne={diagnostics['airborne_time']}, "
            f"contact restored={diagnostics['landing_time']}"
        )
        self.flip_diagnostics = None

    def step(self):
        observation, projected_gravity = self.build_observation()
        self.safety_check(projected_gravity)

        if self.remote.is_pressed(KeyMap.A):
            ready, reason = self._trigger_ready(projected_gravity)
            if ready:
                self.phase_step = 0
                self.backflip_active = True
                print(f"[A] BACKFLIP triggered at control step {self.counter}")
                # Rebuild so this inference sees phase zero immediately.
                observation, projected_gravity = self.build_observation()
                self._start_flip_diagnostics()
            else:
                print(f"[A] trigger rejected: {reason}")

        inference_start = time.perf_counter()
        next_action = self.policy.act(observation)
        self.inference_time_ms = (
            time.perf_counter() - inference_start
        ) * 1000.0
        next_action = np.clip(
            next_action, -self.config.clip_actions, self.config.clip_actions
        ).astype(np.float32)

        # The trained environment applies self.last_actions for the next 20 ms.
        applied_action = self.current_action.copy()
        self.last_action = self.current_action.copy()
        self.current_action = next_action
        max_action_delta = (
            self.config.motor_velocity_limits
            * self.config.target_velocity_fraction
            * self.config.control_dt
            / self.config.action_scale
        )
        self.slew_limited_action += np.clip(
            applied_action - self.slew_limited_action,
            -max_action_delta,
            max_action_delta,
        )
        target = self.config.default_angles + (
            self.config.action_scale * self.slew_limited_action
        )
        # Match training: target positions are not clipped or range-checked.
        # Isaac Gym clips the resulting motor torque instead.
        self._validate_action(next_action)

        elapsed = time.monotonic() - self.policy_enabled_at
        if elapsed < self.config.blend_in_s:
            alpha = elapsed / self.config.blend_in_s
            target = self.enable_pose * (1.0 - alpha) + target * alpha
        self._send_position(target)
        self._update_flip_diagnostics(projected_gravity, target, next_action)

        if self.backflip_active:
            self.phase_step += 1
            if self.phase_step >= self.phase_steps:
                self.phase_step = self.phase_steps
                self.backflip_active = False
                print("[BACKFLIP] phase complete; RL continues landing/recovery")

        if self.verbose or self.counter % 25 == 0:
            phase_time = self.phase_step * self.config.control_dt
            tilt = self._tilt(projected_gravity)
            estimated_torque = (
                self.config.kps * (target - self.q) - self.config.kds * self.dq
            )
            torque_ratio = float(
                np.max(np.abs(estimated_torque) / self.config.torque_limits)
            )
            mode = "FLIP" if self.backflip_active else "WAIT"
            print(
                f"step {self.counter:6d} | {mode} | phase {phase_time:4.2f}s | "
                f"tilt {tilt:4.2f} | maxdq {np.max(np.abs(self.dq)):5.2f} | "
                f"estimated torque ratio {torque_ratio:4.2f}",
                flush=True,
            )
        self.counter += 1

    def damping_stop(self, duration=1.0):
        print(f"[STOP] sending damping commands for {duration:.1f}s")
        count = max(1, int(duration / self.config.control_dt))
        for _ in range(count):
            with self.command_lock:
                self._set_damping_command_locked()
            self.send_cmd()
            time.sleep(self.config.control_dt)


def release_builtin_motion_service():
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )

    client = MotionSwitcherClient()
    client.SetTimeout(5.0)
    client.Init()
    print("[SPORT] releasing built-in motion service")
    for _ in range(5):
        try:
            _, result = client.CheckMode()
        except Exception:
            result = None
        if not result or not result.get("name"):
            return
        print(f"[SPORT] active mode: {result['name']}")
        client.ReleaseMode()
        time.sleep(1.0)
    raise RuntimeError("无法释放内置运动服务")


def check_only(config):
    policy = BackflipPolicy(config.actor_path)
    observation = np.zeros(60, dtype=np.float32)
    observation[5] = -1.0
    observation[[55, 57, 59]] = 1.0  # phase=0 cosine features
    action = policy.act(observation)
    if action.shape != (12,) or not np.all(np.isfinite(action)):
        raise ValueError("ONNX check inference failed")
    print(f"config: {config.file_path}")
    print(f"actor : {policy.path}")
    print(f"input : observation[60]")
    print(f"output: action[12], max |action|={np.max(np.abs(action)):.4f}")
    print("deployment check: OK (no robot connection was opened)")


def main():
    default_config = os.path.join(SCRIPT_DIR, "configs", "go2_backflip.yaml")
    parser = argparse.ArgumentParser(description="Go2 backflip real deployment")
    parser.add_argument("net", nargs="?", help="robot network interface, e.g. enp2s0")
    parser.add_argument("config", nargs="?", default=default_config)
    parser.add_argument("--onnx", default=None, help="override actor_path in YAML")
    parser.add_argument("--check", action="store_true", help="validate config/ONNX only")
    parser.add_argument(
        "--arm-immediately", action="store_true",
        help="skip START arming wait (unsafe; intended only for a secured test stand)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = BackflipConfig(args.config, actor_override=args.onnx)
    if args.check:
        check_only(config)
        return
    if not args.net:
        parser.error("真机运行必须提供网卡名；离线检查请使用 --check")

    policy = BackflipPolicy(config.actor_path)
    sdk = load_unitree_sdk()
    sdk.ChannelFactoryInitialize(0, args.net)
    release_builtin_motion_service()

    controller = BackflipController(config, policy, sdk, verbose=args.verbose)
    stop_reason = "normal exit"
    try:
        controller.start_lowcmd_writer()
        controller.wait_for_low_state()
        controller.startup()
        if not args.arm_immediately:
            controller.wait_for_arm()
        controller.enable_policy()
        print("[CONTROL] A=backflip | SELECT=emergency damping stop")

        next_deadline = time.perf_counter()
        while True:
            if controller.remote.is_down(KeyMap.select):
                stop_reason = "SELECT pressed"
                break
            controller.step()
            next_deadline += config.control_dt
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
                controller.consecutive_late_steps = 0
            else:
                controller.late_steps += 1
                controller.consecutive_late_steps += 1
                controller.max_period = max(controller.max_period, -remaining)
                next_deadline = time.perf_counter()
                if (
                    controller.consecutive_late_steps
                    >= config.max_consecutive_late_steps
                ):
                    raise SafetyStop(
                        f"连续 {controller.consecutive_late_steps} 个控制周期超时"
                    )
    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt"
    except (SafetyStop, TimeoutError) as exc:
        stop_reason = f"safety: {exc}"
    finally:
        try:
            controller.damping_stop()
        finally:
            controller.print_flip_diagnostics_summary()
            controller.stop_lowcmd_writer()

    print(f"Exit: {stop_reason}")
    print(
        f"Loop: late={controller.late_steps}, "
        f"worst lateness={controller.max_period * 1000.0:.1f} ms"
    )
    print(
        f"LowCmd: sent={controller.lowcmd_publish_count}, "
        f"late={controller.lowcmd_late_count}, worst lateness="
        f"{controller.lowcmd_worst_lateness * 1000.0:.2f} ms"
    )


if __name__ == "__main__":
    main()
