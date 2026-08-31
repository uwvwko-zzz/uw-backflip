#!/usr/bin/env python3
"""Run the 60-dimensional Go2 backflip actor continuously in MuJoCo.

The policy owns the complete trajectory. SPACE only restarts the policy phase;
it never teleports or resets the robot. After the two-second backflip phase the
phase is frozen at its final value while the actor keeps recovering/standing.
Only R performs an explicit physical reset.
"""

import argparse
import os
import threading
import time

import mujoco
import numpy as np
import onnxruntime as ort


# Isaac Gym asset DOF order. ONNX actions and observations use this exact order.
JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

DEFAULT_DOF_POS = np.array(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
     0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=np.float32,
)

OBS_DIM = 60
ACTION_DIM = 12
SIM_DT = 0.005
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION
PHASE_DURATION = 2.0
ACTION_SCALE = 0.5
JOINT_VELOCITY_LIMITS = np.full(ACTION_DIM, 30.0, dtype=np.float32)
TARGET_VELOCITY_LIMIT = 13.5
MOTOR_VELOCITY_X1 = 13.5
MOTOR_VELOCITY_X2 = 30.0
MOTOR_TORQUE_Y1 = 20.2
MOTOR_TORQUE_Y2 = 23.4
KP = 40.0
KD = 1.0
JOINT_PASSIVE_DAMPING = 0.0
JOINT_FRICTION_LOSS = 0.0
JOINT_ARMATURE = 0.0
CONTACT_FRICTION = 1.0
OBS_CLIP = 100.0
ACTION_CLIP = 100.0


def expanded_path(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def default_mjcf_path():
    project_model = os.path.join(os.path.dirname(__file__), "go2_isaac.xml")
    candidates = (
        os.environ.get("GO2_MJCF"),
        project_model,
        "~/桌面/mujoco_menagerie/unitree_go2/scene.xml",
        "~/mujoco_menagerie/unitree_go2/scene.xml",
    )
    for candidate in candidates:
        if candidate and os.path.isfile(expanded_path(candidate)):
            return expanded_path(candidate)
    return expanded_path(project_model)


def make_parser():
    parser = argparse.ArgumentParser(
        description="MuJoCo Go2 后空翻 sim2sim（60 维 actor）"
    )
    parser.add_argument("--onnx", required=True, help="pt2onnx.py 导出的 actor ONNX")
    parser.add_argument(
        "--mjcf", default=default_mjcf_path(),
        help="Go2 MJCF；默认使用由训练 URDF 转换的 mujoco/go2/go2_isaac.xml",
    )
    parser.add_argument(
        "--auto-backflip-period", "--auto-jump-period",
        dest="auto_backflip_period", type=float, default=0.0,
        help="自动触发间隔 [s]；0 表示只按 SPACE 触发",
    )
    parser.add_argument(
        "--start", action="store_true", help="启动后立即触发第一次后空翻"
    )
    parser.add_argument(
        "--duration", type=float, default=0.0, help="运行秒数；0 表示持续运行"
    )
    parser.add_argument("--headless", action="store_true", help="不打开 MuJoCo Viewer")
    parser.add_argument("--no-realtime", action="store_true", help="不按真实时间限速")
    return parser


def load_onnx(path):
    path = expanded_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 ONNX 模型: {path}")

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        raise ValueError("后空翻 actor 必须只有一个输入，并至少包含一个输出")

    input_shape = inputs[0].shape
    output_shape = outputs[0].shape
    if input_shape[-1] != OBS_DIM:
        raise ValueError(
            f"ONNX 输入末维应为 {OBS_DIM}，实际为 {input_shape[-1]}。"
            "请使用新版 mujoco/script/pt2onnx.py 导出。"
        )
    if output_shape[-1] != ACTION_DIM:
        raise ValueError(
            f"ONNX 输出末维应为 {ACTION_DIM}，实际为 {output_shape[-1]}"
        )
    return session, inputs[0].name, outputs[0].name, path


def joint_addresses(model):
    qpos_addresses = []
    dof_addresses = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MJCF 中缺少关节: {name}")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def actuator_order(model):
    """Return MuJoCo actuator indices in the Isaac Gym DOF order."""
    indices = []
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(matches) != 1:
            raise ValueError(f"关节 {joint_name} 没有唯一对应的 actuator")
        indices.append(matches[0])
    return np.asarray(indices)


def align_model_parameters(model, dof_addresses):
    """Remove menagerie-only dynamics and match the Isaac Gym asset settings."""
    # The training URDF has no joint <dynamics>, and Isaac asset.armature is 0.
    # Menagerie otherwise adds damping=2, frictionloss=0.2 and armature=0.01.
    model.dof_damping[dof_addresses] = JOINT_PASSIVE_DAMPING
    model.dof_frictionloss[dof_addresses] = JOINT_FRICTION_LOSS
    model.dof_armature[dof_addresses] = JOINT_ARMATURE

    # Isaac evaluation uses its non-randomized rigid-shape friction of 1.0.
    # MuJoCo's torsional/rolling terms do not have matching training parameters,
    # so disable them instead of retaining menagerie-specific resistance.
    collision_geoms = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_friction[collision_geoms, 0] = CONTACT_FRICTION
    model.geom_friction[collision_geoms, 1:] = 0.0


def reset_robot(model, data, qpos_addresses):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = (0.0, 0.0, 0.32)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # MuJoCo quaternion: wxyz
    data.qpos[qpos_addresses] = DEFAULT_DOF_POS
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def projected_gravity_and_body_ang_vel(data, base_body_id):
    # xmat maps body vectors to world; transpose maps world vectors to body.
    body_to_world = data.xmat[base_body_id].reshape(3, 3)
    world_to_body = body_to_world.T
    gravity = world_to_body @ np.array([0.0, 0.0, -1.0])
    # MuJoCo free-joint rotational qvel is already expressed in the child
    # body's local frame. Isaac root angular velocity is world-frame and is
    # rotated once in Go2Backflip; rotating qvel again here was incorrect.
    body_ang_vel = data.qvel[3:6]
    return gravity.astype(np.float32), body_ang_vel.astype(np.float32)


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


def make_observation(
    data,
    base_body_id,
    qpos_addresses,
    dof_addresses,
    action,
    last_action,
    phase_time,
):
    gravity, body_ang_vel = projected_gravity_and_body_ang_vel(
        data, base_body_id
    )
    dof_pos = data.qpos[qpos_addresses].astype(np.float32)
    dof_vel = data.qvel[dof_addresses].astype(np.float32)

    # Must exactly match Go2Backflip.compute_observations().
    observation = np.concatenate(
        (
            body_ang_vel * 0.25,
            gravity,
            dof_pos - DEFAULT_DOF_POS,
            dof_vel * 0.05,
            action,
            last_action,
            phase_features(phase_time),
        )
    ).astype(np.float32)
    if observation.shape != (OBS_DIM,):
        raise RuntimeError(f"观测维度错误: {observation.shape}，预期 ({OBS_DIM},)")
    return np.clip(observation, -OBS_CLIP, OBS_CLIP)


class KeyboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._trigger_requested = False
        self._reset_requested = False
        self._space_down = False
        self.running = True

    def on_press(self, key):
        key_name = self._key_name(key)
        with self._lock:
            if key_name == "space" and not self._space_down:
                self._trigger_requested = True
                self._space_down = True
            elif key_name == "r":
                self._reset_requested = True
            elif key_name == "esc":
                self.running = False

    def on_release(self, key):
        key_name = self._key_name(key)
        with self._lock:
            if key_name == "space":
                self._space_down = False
            elif key_name == "esc":
                self.running = False
                return False
        return None

    @staticmethod
    def _key_name(key):
        char = getattr(key, "char", None)
        if char:
            return char.lower()
        name = getattr(key, "name", None)
        return name.lower() if name else str(key).lower()

    def consume(self):
        with self._lock:
            trigger = self._trigger_requested
            reset = self._reset_requested
            self._trigger_requested = False
            self._reset_requested = False
        return trigger, reset


class _NullViewer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def is_running(self):
        return True

    def sync(self):
        pass


def main():
    args = make_parser().parse_args()
    if args.auto_backflip_period < 0.0:
        raise ValueError("--auto-backflip-period 不能小于 0")

    onnx_session, input_name, output_name, onnx_path = load_onnx(args.onnx)
    mjcf_path = expanded_path(args.mjcf)
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(
            f"找不到 Go2 MJCF: {mjcf_path}\n"
            "请通过 --mjcf=/path/to/scene.xml 指定。"
        )

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    qpos_addresses, dof_addresses = joint_addresses(model)
    actuator_indices = actuator_order(model)
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ])
    joint_ranges = model.jnt_range[joint_ids]
    align_model_parameters(model, dof_addresses)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_body_id < 0:
        raise ValueError("MJCF 中缺少 base 刚体")

    reset_robot(model, data, qpos_addresses)
    keyboard_state = KeyboardState()
    listener = None
    if not args.headless:
        # Importing pynput connects to X, so keep it out of headless mode.
        from pynput import keyboard

        listener = keyboard.Listener(
            on_press=keyboard_state.on_press,
            on_release=keyboard_state.on_release,
        )
        listener.start()

    current_action = np.zeros(ACTION_DIM, dtype=np.float32)
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    slew_limited_action = np.zeros(ACTION_DIM, dtype=np.float32)
    phase_time = 0.0
    backflip_active = bool(args.start)
    policy_step = 0
    next_auto_trigger = (
        args.auto_backflip_period if args.auto_backflip_period > 0.0 else np.inf
    )
    head_body_ids = {
        body_id for body_id in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        .lower().startswith("head")
    }
    cumulative_pitch = 0.0
    max_abs_joint_speed = 0.0
    max_speed_joint = ""
    max_joint_limit_excess = 0.0
    head_contact = False
    min_base_height = float(data.qpos[2])
    max_base_height = float(data.qpos[2])

    print("=" * 72)
    print(f"ONNX           : {onnx_path}")
    print(f"MJCF           : {mjcf_path}")
    print(f"Policy input   : observation[{OBS_DIM}]")
    print(f"Control        : {1.0 / POLICY_DT:.0f} Hz, Kp/Kd={KP:g}/{KD:g}, scale={ACTION_SCALE:g}")
    print(f"Action latency : one policy step ({POLICY_DT * 1000.0:.0f} ms)")
    print(
        "Joint dynamics : "
        f"damping={JOINT_PASSIVE_DAMPING:g}, "
        f"frictionloss={JOINT_FRICTION_LOSS:g}, "
        f"armature={JOINT_ARMATURE:g}"
    )
    print(f"Contact friction: {CONTACT_FRICTION:g}")
    print("Controls       : SPACE backflip | R physical reset | ESC quit")
    print("No automatic physical reset: RL controls takeoff, landing and recovery.")
    print("=" * 72)
    if backflip_active:
        print("[START] backflip triggered")

    viewer_context = (
        _NullViewer() if args.headless else mujoco.viewer.launch_passive(model, data)
    )
    wall_start = time.perf_counter()
    sim_start = data.time

    try:
        with viewer_context as viewer:
            while viewer.is_running() and keyboard_state.running:
                if args.duration > 0.0 and data.time - sim_start >= args.duration:
                    break

                manual_trigger, reset_requested = keyboard_state.consume()
                if reset_requested:
                    reset_robot(model, data, qpos_addresses)
                    current_action.fill(0.0)
                    last_action.fill(0.0)
                    slew_limited_action.fill(0.0)
                    phase_time = 0.0
                    backflip_active = False
                    print("[R] robot, phase and action history reset")

                auto_trigger = False
                while data.time >= next_auto_trigger:
                    auto_trigger = True
                    next_auto_trigger += args.auto_backflip_period
                if (manual_trigger or auto_trigger) and not backflip_active:
                    # Phase reset only. Robot state and action history stay continuous.
                    phase_time = 0.0
                    backflip_active = True
                    source = "SPACE" if manual_trigger else "AUTO"
                    print(f"[{source}] backflip triggered at sim t={data.time:.2f}s")

                observation = make_observation(
                    data, base_body_id, qpos_addresses, dof_addresses,
                    current_action, last_action, phase_time,
                )
                next_action = onnx_session.run(
                    [output_name],
                    {input_name: observation[None, :].astype(np.float32)},
                )[0][0]
                if not np.all(np.isfinite(next_action)):
                    raise FloatingPointError(f"ONNX 输出包含 NaN/Inf: {next_action}")
                next_action = np.clip(
                    next_action, -ACTION_CLIP, ACTION_CLIP
                ).astype(np.float32)

                # Training uses last_actions for PD torque: fixed 20-ms latency.
                applied_action = current_action.copy()
                last_action = current_action.copy()
                current_action = next_action

                for _ in range(DECIMATION):
                    step_wall_start = time.perf_counter()
                    max_action_delta = (
                        TARGET_VELOCITY_LIMIT
                        * SIM_DT
                        / ACTION_SCALE
                    )
                    slew_limited_action += np.clip(
                        applied_action - slew_limited_action,
                        -max_action_delta,
                        max_action_delta,
                    )
                    target_pos = (
                        DEFAULT_DOF_POS + ACTION_SCALE * slew_limited_action
                    )
                    dof_pos = data.qpos[qpos_addresses]
                    dof_vel = data.qvel[dof_addresses]
                    torque = KP * (target_pos - dof_pos) - KD * dof_vel
                    speed = np.abs(dof_vel)
                    speed_fraction = np.where(
                        speed < MOTOR_VELOCITY_X1,
                        1.0,
                        np.clip(
                            (MOTOR_VELOCITY_X2 - speed)
                            / (MOTOR_VELOCITY_X2 - MOTOR_VELOCITY_X1),
                            0.0,
                            1.0,
                        ),
                    )
                    peak_torque = np.where(
                        dof_vel * torque > 0.0,
                        MOTOR_TORQUE_Y1,
                        MOTOR_TORQUE_Y2,
                    )
                    torque_limit = peak_torque * speed_fraction
                    data.ctrl[actuator_indices] = np.clip(
                        torque, -torque_limit, torque_limit
                    )
                    mujoco.mj_step(model, data)
                    step_dof_vel = data.qvel[dof_addresses]
                    speed_index = int(np.argmax(np.abs(step_dof_vel)))
                    step_max_speed = float(np.abs(step_dof_vel[speed_index]))
                    if step_max_speed > max_abs_joint_speed:
                        max_abs_joint_speed = step_max_speed
                        max_speed_joint = JOINT_NAMES[speed_index]
                    step_dof_pos = data.qpos[qpos_addresses]
                    limit_excess = np.maximum(
                        joint_ranges[:, 0] - step_dof_pos,
                        step_dof_pos - joint_ranges[:, 1],
                    )
                    max_joint_limit_excess = max(
                        max_joint_limit_excess,
                        float(np.max(np.maximum(limit_excess, 0.0))),
                    )
                    cumulative_pitch += float(data.qvel[4]) * SIM_DT
                    min_base_height = min(min_base_height, float(data.qpos[2]))
                    max_base_height = max(max_base_height, float(data.qpos[2]))
                    for contact_index in range(data.ncon):
                        contact = data.contact[contact_index]
                        body1 = int(model.geom_bodyid[contact.geom1])
                        body2 = int(model.geom_bodyid[contact.geom2])
                        if body1 in head_body_ids or body2 in head_body_ids:
                            head_contact = True
                    if not np.all(np.isfinite(data.qpos)):
                        raise FloatingPointError("MuJoCo 状态包含 NaN/Inf，仿真已发散")
                    viewer.sync()
                    if not args.no_realtime:
                        elapsed = time.perf_counter() - step_wall_start
                        if elapsed < SIM_DT:
                            time.sleep(SIM_DT - elapsed)

                if backflip_active:
                    phase_time = min(phase_time + POLICY_DT, PHASE_DURATION)
                    if phase_time >= PHASE_DURATION:
                        backflip_active = False
                        print(
                            "[BACKFLIP] phase complete; actor remains in control "
                            "for landing/recovery"
                        )

                if policy_step % 10 == 0:
                    gravity, body_ang_vel = projected_gravity_and_body_ang_vel(
                        data, base_body_id
                    )
                    del gravity
                    mode = "FLIP" if backflip_active else "WAIT"
                    print(
                        f"step {policy_step:6d} | {mode:4s} | "
                        f"phase {phase_time:4.2f}s | base_z {data.qpos[2]:5.3f}m | "
                        f"pitch_rate {body_ang_vel[1]:+6.3f}rad/s",
                        flush=True,
                    )
                policy_step += 1
    finally:
        if listener is not None:
            listener.stop()

    wall_elapsed = time.perf_counter() - wall_start
    final_gravity, final_ang_vel = projected_gravity_and_body_ang_vel(
        data, base_body_id
    )
    final_tilt = float(np.arccos(np.clip(-final_gravity[2], -1.0, 1.0)))
    completed_rotation = abs(cumulative_pitch) >= 5.5
    stable_finish = (
        final_tilt < 0.35
        and float(data.qpos[2]) > 0.25
        and abs(float(final_ang_vel[1])) < 1.0
    )
    validation_pass = (
        completed_rotation
        and stable_finish
        and not head_contact
        and max_joint_limit_excess <= 1.0e-3
        and max_abs_joint_speed <= 31.5
    )
    print(
        f"结束：仿真 {data.time - sim_start:.2f}s，墙钟 {wall_elapsed:.2f}s，"
        f"机身高度 {data.qpos[2]:.3f}m"
    )
    print(
        "诊断："
        f"累计俯仰 {cumulative_pitch:+.3f} rad | "
        f"机身高度 {min_base_height:.3f}--{max_base_height:.3f} m | "
        f"最终倾角 {final_tilt:.3f} rad"
    )
    print(
        "诊断："
        f"最大关节速度 {max_abs_joint_speed:.2f} rad/s ({max_speed_joint}) | "
        f"最大限位超出 {max_joint_limit_excess:.4f} rad | "
        f"头部接触 {'是' if head_contact else '否'}"
    )
    print(f"验证：{'PASS' if validation_pass else 'FAIL'}")


if __name__ == "__main__":
    # viewer is not automatically exposed by import mujoco.
    import mujoco.viewer

    main()
