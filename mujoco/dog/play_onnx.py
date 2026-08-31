#!/usr/bin/env python3
"""MuJoCo sim2sim validation for the custom dog 60-D backflip actor."""

import argparse
import os
import threading
import time

import mujoco
import numpy as np
import onnxruntime as ort


JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
DEFAULT_DOF_POS = np.asarray(
    [0.0, -0.8, -1.5, 0.0, 0.8, 1.5,
     0.0, -0.8, -1.5, 0.0, 0.8, 1.5],
    dtype=np.float32,
)
JOINT_RANGES = np.asarray(
    [
        [-1.05, 1.05], [-1.57, 3.49], [-2.72, -1.20],
        [-1.05, 1.05], [-1.57, 3.49], [1.20, 2.72],
        [-1.05, 1.05], [-1.57, 3.49], [-2.72, -1.20],
        [-1.05, 1.05], [-1.57, 3.49], [1.20, 2.72],
    ],
    dtype=np.float64,
)
TORQUE_LIMITS = np.asarray(
    [17.0, 17.0, 34.0] * 4, dtype=np.float64
)
VELOCITY_LIMITS = np.asarray(
    [30.1, 30.1, 20.07] * 4, dtype=np.float64
)
TARGET_VELOCITY_LIMITS = np.full(12, 13.5, dtype=np.float64)
MOTOR_VELOCITY_X1 = np.full(12, 13.5, dtype=np.float64)

OBS_DIM = 60
ACTION_DIM = 12
SIM_DT = 0.005
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION
PHASE_DURATION = 2.0
ACTION_SCALE = 0.5
KP = 40.0
KD = 1.0
BASE_HEIGHT = 0.344
OBS_CLIP = 100.0
ACTION_CLIP = 100.0


def expanded_path(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_mjcf_path():
    return os.path.join(
        project_root(), "resources", "robots", "dog", "xml", "dog_1.xml"
    )


def parser():
    result = argparse.ArgumentParser(
        description="Custom dog 后空翻 MuJoCo sim2sim（60维 actor）"
    )
    result.add_argument("--onnx", required=True, help="actor ONNX模型")
    result.add_argument(
        "--mjcf", default=default_mjcf_path(), help="dog MJCF路径"
    )
    result.add_argument(
        "--start", action="store_true", help="启动后立即触发后空翻"
    )
    result.add_argument(
        "--auto-backflip-period",
        type=float,
        default=0.0,
        help="自动触发周期；0表示只按SPACE触发",
    )
    result.add_argument(
        "--duration", type=float, default=0.0, help="运行秒数；0持续运行"
    )
    result.add_argument("--headless", action="store_true")
    result.add_argument("--no-realtime", action="store_true")
    result.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="MuJoCo足地滑动摩擦系数（默认1.0）",
    )
    result.add_argument(
        "--impratio",
        type=float,
        default=1.0,
        help="MuJoCo摩擦/法向约束阻抗比（默认1.0）",
    )
    return result


def load_onnx(path):
    path = expanded_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到ONNX模型: {path}")
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        raise ValueError("actor必须只有一个输入，并至少包含一个输出")
    if inputs[0].shape[-1] != OBS_DIM:
        raise ValueError(
            f"ONNX输入应为{OBS_DIM}维，实际为{inputs[0].shape}"
        )
    if outputs[0].shape[-1] != ACTION_DIM:
        raise ValueError(
            f"ONNX输出应为{ACTION_DIM}维，实际为{outputs[0].shape}"
        )
    return session, inputs[0].name, outputs[0].name, path


def model_addresses(model):
    joint_ids = []
    qpos_addresses = []
    dof_addresses = []
    actuator_indices = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"MJCF缺少关节: {name}")
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(matches) != 1:
            raise ValueError(f"关节{name}没有唯一对应的motor")
        joint_ids.append(joint_id)
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])
        actuator_indices.append(matches[0])
    return (
        np.asarray(joint_ids),
        np.asarray(qpos_addresses),
        np.asarray(dof_addresses),
        np.asarray(actuator_indices),
    )




def align_model(
    model, joint_ids, dof_addresses, actuator_indices, contact_friction
):
    model.opt.timestep = SIM_DT
    model.dof_damping[dof_addresses] = 0.0
    model.dof_frictionloss[dof_addresses] = 0.0
    model.dof_armature[dof_addresses] = 0.0
    model.jnt_limited[joint_ids] = 1
    model.jnt_range[joint_ids] = JOINT_RANGES
    model.actuator_ctrllimited[actuator_indices] = 1
    model.actuator_ctrlrange[actuator_indices, 0] = -TORQUE_LIMITS
    model.actuator_ctrlrange[actuator_indices, 1] = TORQUE_LIMITS
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_friction[collision, 0] = contact_friction
    model.geom_friction[collision, 1:] = 0.0
    # Isaac asset.self_collisions=1 disables robot self collision. Use
    # complementary floor/robot bitmasks so robot geoms still contact ground.
    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    robot_collision = collision.copy()
    robot_collision[floor_id] = False
    cylinders = robot_collision & (
        model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER
    )
    model.geom_type[cylinders] = mujoco.mjtGeom.mjGEOM_CAPSULE
    model.geom_margin[robot_collision] = 0.01
    model.geom_contype[robot_collision] = 2
    model.geom_conaffinity[robot_collision] = 1
    model.geom_contype[floor_id] = 1
    model.geom_conaffinity[floor_id] = 2


def find_base(model):
    for name in ("trunk", "base"):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id >= 0:
            return body_id, name
    raise ValueError("MJCF中找不到trunk/base刚体")


def reset_robot(model, data, qpos_addresses):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = (0.0, 0.0, BASE_HEIGHT)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[qpos_addresses] = DEFAULT_DOF_POS
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def gravity_and_ang_vel(data, base_body_id):
    body_to_world = data.xmat[base_body_id].reshape(3, 3)
    gravity = body_to_world.T @ np.asarray([0.0, 0.0, -1.0])
    # MuJoCo free-joint rotational velocity uses the child-body frame.
    return gravity.astype(np.float32), data.qvel[3:6].astype(np.float32)


def phase_features(phase_time):
    phase = np.pi * phase_time / 2.0
    return np.asarray(
        [
            np.sin(phase), np.cos(phase),
            np.sin(phase / 2.0), np.cos(phase / 2.0),
            np.sin(phase / 4.0), np.cos(phase / 4.0),
        ],
        dtype=np.float32,
    )


def observation(
    data,
    base_body_id,
    qpos_addresses,
    dof_addresses,
    action,
    last_action,
    phase_time,
):
    gravity, ang_vel = gravity_and_ang_vel(data, base_body_id)
    value = np.concatenate(
        (
            ang_vel * 0.25,
            gravity,
            data.qpos[qpos_addresses].astype(np.float32) - DEFAULT_DOF_POS,
            data.qvel[dof_addresses].astype(np.float32) * 0.05,
            action,
            last_action,
            phase_features(phase_time),
        )
    ).astype(np.float32)
    if value.shape != (OBS_DIM,):
        raise RuntimeError(f"观测维度错误: {value.shape}")
    return np.clip(value, -OBS_CLIP, OBS_CLIP)


class Keyboard:
    def __init__(self):
        self.lock = threading.Lock()
        self.trigger = False
        self.reset = False
        self.space_down = False
        self.running = True

    @staticmethod
    def name(key):
        char = getattr(key, "char", None)
        if char:
            return char.lower()
        return (getattr(key, "name", None) or str(key)).lower()

    def on_press(self, key):
        name = self.name(key)
        with self.lock:
            if name == "space" and not self.space_down:
                self.trigger = True
                self.space_down = True
            elif name == "r":
                self.reset = True
            elif name == "esc":
                self.running = False

    def on_release(self, key):
        with self.lock:
            if self.name(key) == "space":
                self.space_down = False

    def consume(self):
        with self.lock:
            trigger, reset = self.trigger, self.reset
            self.trigger = False
            self.reset = False
            return trigger, reset


class NullViewer:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def is_running(self):
        return True

    def sync(self):
        pass


def main():
    args = parser().parse_args()
    if args.auto_backflip_period < 0.0:
        raise ValueError("--auto-backflip-period不能小于0")
    if args.friction <= 0.0:
        raise ValueError("--friction必须大于0")
    if args.impratio <= 0.0:
        raise ValueError("--impratio必须大于0")
    session, input_name, output_name, onnx_path = load_onnx(args.onnx)
    mjcf_path = expanded_path(args.mjcf)
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(f"找不到dog MJCF: {mjcf_path}")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    model.opt.impratio = args.impratio
    data = mujoco.MjData(model)
    joint_ids, qpos_addresses, dof_addresses, actuator_indices = (
        model_addresses(model)
    )
    align_model(
        model, joint_ids, dof_addresses, actuator_indices, args.friction
    )
    base_body_id, base_name = find_base(model)
    reset_robot(model, data, qpos_addresses)

    keyboard = Keyboard()
    listener = None
    if not args.headless:
        from pynput import keyboard as pynput_keyboard

        listener = pynput_keyboard.Listener(
            on_press=keyboard.on_press, on_release=keyboard.on_release
        )
        listener.start()

    current_action = np.zeros(ACTION_DIM, dtype=np.float32)
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    slew_action = np.zeros(ACTION_DIM, dtype=np.float32)
    phase_time = 0.0
    active = bool(args.start)
    next_auto = (
        args.auto_backflip_period
        if args.auto_backflip_period > 0.0
        else np.inf
    )
    policy_step = 0
    cumulative_backflip = 0.0
    max_joint_speed = np.zeros(ACTION_DIM)
    max_limit_excess = 0.0
    trunk_contact = False
    min_height = float(data.qpos[2])
    max_height = float(data.qpos[2])
    initial_x = float(data.qpos[0])
    min_x = initial_x
    max_x = initial_x

    print("=" * 76)
    print(f"ONNX            : {onnx_path}")
    print(f"MJCF            : {mjcf_path}")
    print(f"Base body       : {base_name}")
    print(f"Model mass      : {model.body_mass.sum():.5f} kg")
    print(f"Policy          : {OBS_DIM} -> {ACTION_DIM}, 50 Hz")
    print(f"PD              : Kp/Kd={KP:g}/{KD:g}, action scale={ACTION_SCALE:g}")
    print("Action latency  : fixed 20 ms")
    print("Torque limits   : hip/thigh 17 Nm, calf 34 Nm")
    print(f"Contact friction: {args.friction:g}")
    print(f"Friction impratio: {args.impratio:g}")
    print("Controls        : SPACE backflip | R reset | ESC quit")
    print("No automatic reset: RL owns takeoff, landing and recovery.")
    print("=" * 76)
    if active:
        print("[START] backflip triggered")

    if args.headless:
        viewer_context = NullViewer()
    else:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(model, data)

    sim_start = data.time
    wall_start = time.perf_counter()
    try:
        with viewer_context as viewer:
            while viewer.is_running() and keyboard.running:
                if args.duration > 0.0 and data.time - sim_start >= args.duration:
                    break
                manual_trigger, reset_requested = keyboard.consume()
                if reset_requested:
                    reset_robot(model, data, qpos_addresses)
                    current_action.fill(0.0)
                    last_action.fill(0.0)
                    slew_action.fill(0.0)
                    phase_time = 0.0
                    active = False
                    print("[R] physical state, phase and history reset")

                auto_trigger = False
                while data.time >= next_auto:
                    auto_trigger = True
                    next_auto += args.auto_backflip_period
                if (manual_trigger or auto_trigger) and not active:
                    phase_time = 0.0
                    active = True
                    print(
                        f"[{'SPACE' if manual_trigger else 'AUTO'}] "
                        f"backflip triggered at {data.time:.2f}s"
                    )

                obs = observation(
                    data, base_body_id, qpos_addresses, dof_addresses,
                    current_action, last_action, phase_time,
                )
                next_action = session.run(
                    [output_name], {input_name: obs[None, :]}
                )[0][0]
                if not np.all(np.isfinite(next_action)):
                    raise FloatingPointError("ONNX输出包含NaN/Inf")
                next_action = np.clip(
                    next_action, -ACTION_CLIP, ACTION_CLIP
                ).astype(np.float32)

                # One 50-Hz policy-step action delay, matching nominal play.
                applied_action = current_action.copy()
                last_action = current_action.copy()
                current_action = next_action

                for _ in range(DECIMATION):
                    tick = time.perf_counter()
                    max_delta = (
                        TARGET_VELOCITY_LIMITS * SIM_DT / ACTION_SCALE
                    )
                    slew_action += np.clip(
                        applied_action - slew_action, -max_delta, max_delta
                    )
                    target = DEFAULT_DOF_POS + ACTION_SCALE * slew_action
                    q = data.qpos[qpos_addresses]
                    dq = data.qvel[dof_addresses]
                    torque = KP * (target - q) - KD * dq
                    speed = np.abs(dq)
                    speed_fraction = np.where(
                        speed < MOTOR_VELOCITY_X1,
                        1.0,
                        np.clip(
                            (VELOCITY_LIMITS - speed)
                            / (VELOCITY_LIMITS - MOTOR_VELOCITY_X1),
                            0.0,
                            1.0,
                        ),
                    )
                    limit = TORQUE_LIMITS * speed_fraction
                    data.ctrl[actuator_indices] = np.clip(
                        torque, -limit, limit
                    )
                    mujoco.mj_step(model, data)
                    dq = data.qvel[dof_addresses]
                    max_joint_speed = np.maximum(max_joint_speed, np.abs(dq))
                    q = data.qpos[qpos_addresses]
                    excess = np.maximum(
                        JOINT_RANGES[:, 0] - q, q - JOINT_RANGES[:, 1]
                    )
                    max_limit_excess = max(
                        max_limit_excess,
                        float(np.maximum(excess, 0.0).max()),
                    )
                    _, ang_vel = gravity_and_ang_vel(data, base_body_id)
                    cumulative_backflip += -float(ang_vel[1]) * SIM_DT
                    min_height = min(min_height, float(data.qpos[2]))
                    max_height = max(max_height, float(data.qpos[2]))
                    min_x = min(min_x, float(data.qpos[0]))
                    max_x = max(max_x, float(data.qpos[0]))
                    for contact_index in range(data.ncon):
                        contact = data.contact[contact_index]
                        body1 = model.geom_bodyid[contact.geom1]
                        body2 = model.geom_bodyid[contact.geom2]
                        if body1 == base_body_id or body2 == base_body_id:
                            trunk_contact = True
                    if not np.all(np.isfinite(data.qpos)):
                        raise FloatingPointError("MuJoCo状态发散")
                    viewer.sync()
                    if not args.no_realtime:
                        elapsed = time.perf_counter() - tick
                        if elapsed < SIM_DT:
                            time.sleep(SIM_DT - elapsed)

                if active:
                    phase_time = min(
                        phase_time + POLICY_DT, PHASE_DURATION
                    )
                    if phase_time >= PHASE_DURATION:
                        active = False
                        print(
                            "[BACKFLIP] phase complete; RL continues recovery"
                        )

                if policy_step % 10 == 0:
                    _, ang_vel = gravity_and_ang_vel(data, base_body_id)
                    print(
                        f"step {policy_step:5d} | "
                        f"{'FLIP' if active else 'WAIT':4s} | "
                        f"phase {phase_time:4.2f}s | "
                        f"x {data.qpos[0]:+6.3f}m | "
                        f"vx {data.qvel[0]:+6.3f}m/s | "
                        f"z {data.qpos[2]:5.3f}m | "
                        f"pitch_rate {ang_vel[1]:+6.3f}rad/s",
                        flush=True,
                    )
                policy_step += 1
    finally:
        if listener is not None:
            listener.stop()

    gravity, ang_vel = gravity_and_ang_vel(data, base_body_id)
    tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
    speed_ratio = max_joint_speed / VELOCITY_LIMITS
    max_speed_index = int(np.argmax(speed_ratio))
    completed = cumulative_backflip >= 5.5
    stable = (
        tilt < 0.35
        and float(data.qpos[2]) > 0.27
        and abs(float(ang_vel[1])) < 1.0
    )
    motion_pass = completed and stable and not trunk_contact
    safety_pass = (
        max_limit_excess <= 1.0e-3
        and float(speed_ratio.max()) <= 1.05
    )
    print("-" * 76)
    print(
        f"结束：sim={data.time - sim_start:.2f}s，"
        f"wall={time.perf_counter() - wall_start:.2f}s"
    )
    print(
        f"累计后翻角={cumulative_backflip:+.3f}rad | "
        f"高度={min_height:.3f}--{max_height:.3f}m | "
        f"最终倾角={tilt:.3f}rad"
    )
    print(
        f"水平位移x={data.qpos[0] - initial_x:+.3f}m | "
        f"x范围={min_x:+.3f}--{max_x:+.3f}m | "
        f"最终vx={data.qvel[0]:+.3f}m/s"
    )
    print(
        f"最大速度={max_joint_speed[max_speed_index]:.2f}rad/s "
        f"({JOINT_NAMES[max_speed_index]}, "
        f"{speed_ratio[max_speed_index]:.2f}x限值) | "
        f"最大限位超出={max_limit_excess:.4f}rad | "
        f"base接触={'是' if trunk_contact else '否'}"
    )
    print(
        "最终关节角="
        + np.array2string(
            data.qpos[qpos_addresses], precision=3, suppress_small=True
        )
    )
    print(f"动作验证：{'PASS' if motion_pass else 'FAIL'}")
    print(
        "安全检查："
        + (
            "PASS"
            if safety_pass
            else "WARN（动作成功，但MuJoCo记录到超速或关节限位超出）"
        )
    )


if __name__ == "__main__":
    main()
