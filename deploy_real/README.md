# Go2 后空翻 Sim2Real 部署

本目录部署当前 `go2_backflip` 的 60 维 actor。策略从起跳、旋转、落地、恢复到下一次等待全程保持 RL 接管；按键只改变策略相位，不会重置真机状态。

## 文件

```text
deploy_real/
├── deploy_real_backflip.py       # 当前后空翻真机控制器
├── backflip_config.py            # 后空翻配置加载与参数检查
├── configs/go2_backflip.yaml     # 与训练一致的 Go2 参数
└── common/
    ├── command_helper.py         # LowCmd 初始化
    ├── remote_controller.py      # 手柄解析及按键锁存
    └── rotation_helper.py        # projected_gravity
```


## 3. 不连接真机的检查

先验证 YAML、ONNX 输入输出尺寸和一次推理：

```bash
python deploy_real/deploy_real_backflip.py \
  --check \
  --onnx outputs/model_1900_actor.onnx
```

成功时最后显示：

```text
deployment check: OK (no robot connection was opened)
```

这个命令不会加载 Unitree SDK，也不会打开 DDS 或发送电机命令。

## 4. 网络

电脑有线网卡设置为与 Go2 相同网段，例如：

```text
Go2:  192.168.123.10
电脑: 192.168.123.100
```

查询网卡名：

```bash
ip addr
```

下面假设网卡名是 `enp5s0`。

## 5. 真机运行

Go2 已经正常站立时运行：

```bash
python deploy_real/deploy_real_backflip.py \
  enp5s0 \
  deploy_real/configs/go2_backflip.yaml
```

程序流程：

1. 释放宇树内置运动服务。
2. 接收 `LowState`，检查 IMU 和关节状态。
3. 保持当前姿态 1 秒。
4. 用 2 秒余弦曲线进入训练默认姿态。
5. 保持默认姿态并标定足端压力基线。
6. 等待手柄 `START`，不会自动启用 actor。
7. actor 启用后冻结在相位 0，等待触发。

## 部署逻辑

### 状态机

程序不会一启动就执行后空翻，而是依次进入以下状态：

```text
DDS 初始化
    |
    v
接收 LowState + 启动 500 Hz LowCmd 线程
    |
    v
STARTUP：保持当前姿态 -> 平滑进入 default pose -> 标定足端压力
    |
    v
ARM：等待 START，使用电机端位置 PD 保持站姿
    |
    v
WAIT：启用 60 维 actor，phase=0，等待 A
    |
    | A 且满足直立、低关节速度和足端接触
    v
FLIP：phase 从 0 增加到 2.0 s，RL 连续控制
    |
    v
RECOVERY/WAIT：phase 冻结在 2.0 s，RL 继续落地和恢复
    |
    `---- 再次满足触发条件后，可以接受下一次 A

任意状态发生 SELECT、Ctrl-C 或通信/电机看门狗异常
    -> 切换到阻尼命令并持续发送 1 s
```

按 A 只把 `phase_step` 设为 0，并立即使用相位 0 重新构造观测。机器人
位置、速度、IMU、当前动作和上一拍动作都不会清零，因此起跳、旋转、落地、
恢复以及下一次等待是一条连续真实轨迹。

### 50 Hz Actor 主循环

Actor 主循环每 20 ms 执行一次：

1. 从最新 `LowState` 按策略关节顺序读取 `q/dq`、IMU 和手柄状态。
2. 用 IMU 四元数计算 `projected_gravity`，拼接 60 维观测。
3. 检查 LowState 时效、关节速度以及观测中的 NaN/Inf。
4. ONNX 推理得到 `next_action`，检查输出是否为有限数且幅值正常。
5. 把旧 `current_action` 作为本拍实际动作，再更新动作历史，形成与训练一致
   的一拍 20 ms actor 延迟。
6. 对动作做 13.5 rad/s 等效目标变化率限制，然后计算
   `target = default_q + action_scale * action`。
7. 更新线程共享的完整 12 维目标和时间戳。
8. FLIP 状态推进相位；到 2.0 s 后只冻结相位，不关闭 actor。

观测中的 `current_action` 和 `last_action` 是策略动作历史，不是电机反馈。
它们的更新顺序必须和训练一致，否则 ONNX 即使相同，真机输入也会错一拍。

### 500 Hz LowCmd 线程

独立线程每 2 ms 获取最新实测 `q/dq` 和共享目标，计算：

```text
raw_tau     = Kp * (target - q) - Kd * dq
limited_tau = Go2HV_torque_speed_clip(raw_tau, dq)
```

Go2HV 包络在低速使用 20.2 Nm 驱动力、23.4 Nm 制动力；超过
13.5 rad/s 后线性降额，到 30 rad/s 时可用力矩降为零。

RL 接管后直接发送最终限幅力矩，避免宇树电机端再做第二次 PD：

```text
motor.q   = POS_STOP_F
motor.dq  = VEL_STOP_F
motor.kp  = 0
motor.kd  = 0
motor.tau = limited_tau
```

STARTUP 和等待 START 时仍使用电机端位置 PD。三种发布模式为：

| 阶段 | `q` | `Kp/Kd` | `tau` | 用途 |
| --- | --- | --- | --- | --- |
| STARTUP / ARM | 限幅后位置目标 | 40 / 1 | 0 | 平滑站立和武装等待 |
| WAIT / FLIP / RECOVERY | STOP 标志 | 0 / 0 | 动态限幅力矩 | 与训练一致的直力矩控制 |
| STOP | 0 | 0 / `damping_kd` | 0 | 阻尼急停 |

500 Hz 线程还负责 LowCmd CRC、LowState/策略目标超时、电压、电机 `lost`
和温度看门狗。即使 50 Hz Python 主循环卡住，它也可以根据目标超时主动
切换阻尼模式。

### 关节映射与线程同步

策略关节顺序是 `FL, FR, RL, RR`，Unitree 电机顺序是
`FR, FL, RR, RL`。`leg_joint2motor_idx` 在读取 LowState 和写入 LowCmd
时执行同一映射，保证训练、ONNX 与真机电机编号一致。

50 Hz 主循环和 500 Hz 线程通过锁保护的共享目标通信。主循环一次性更新
完整的目标、Kp/Kd 和时间戳；发布线程读取同一快照后写出完整 LowCmd，
避免出现部分关节使用新命令、部分关节仍使用旧命令的竞争状态。

## 手柄

```text
START   武装并启用 RL
A       触发一次后空翻
SELECT  紧急进入阻尼模式
Ctrl-C  紧急进入阻尼模式
```

按 A 时只执行：

```text
phase = 0
```

机器人位置、速度、actor 动作历史不会被清零。2 秒相位结束后，相位冻结在 2 秒，actor 继续负责落地和恢复。机器人直立、关节速度足够低且足端有接触时，下一次 A 才会被接受。

## 与训练一致的推理链

60 维观测：

```text
body_ang_vel * 0.25       3
projected_gravity         3
q - default_q            12
dq * 0.05                12
current_action           12
last_action              12
phase features            6
                           ──
                          60
```

相位特征：

```text
phi = pi * phase_time / 2
sin(phi), cos(phi), sin(phi/2), cos(phi/2), sin(phi/4), cos(phi/4)
```

控制参数：

```text
policy frequency   50 Hz
LowCmd publish    500 Hz
phase duration      2.0 s
Kp / Kd            40 / 1
action scale        0.5
action latency      1 policy step = 20 ms
```

Actor 每 20 ms 更新一次目标，独立 LowCmd 线程每 2 ms 重发最新的完整
`q/dq/Kp/Kd/tau` 电机命令。`dq` 是 Unitree DDS 消息的真实字段名；
不要写成不会被序列化的 `qd`。

500 Hz 层用最新实测 `q/dq` 计算 PD 力矩，再应用与训练相同的 Go2HV
曲线（13.5 rad/s 前保持 20.2/23.4 Nm 驱动/制动能力，30 rad/s 降为
零）。RL 阶段通过 `MotorCmd.tau` 发送最终限幅力矩，不再反算位置命令，
也不会让宇树电机端重复计算第二次 PD。

## 跳跃诊断

按 A 后自动记录 5 秒。终端每 0.1 秒输出一行紧凑状态，每 0.5 秒输出
一次完整向量，包含：

- IMU 三轴角速度、机身倾角与 LowState 消息延迟；
- 最大关节速度、目标位置误差、原始 PD 扭矩需求及额定扭矩倍率；
- 电机 `tau_est`、温度、lost 计数；
- 累计后翻角、离地/重新接触时间、每拍扭矩饱和关节数；
- `tau_cmd - tau_est` 跟踪误差，以及500 Hz线程捕获的电源峰值；
- actor 动作、目标关节角、四足压力；
- 电池电压/电流和 ONNX 推理时间。

捕获结束或 Ctrl-C 时会输出峰值、对应关节和触发后的时间。程序退出还会
单独报告 50 Hz actor 循环与 500 Hz LowCmd 循环的迟到次数。

目标角不做关节限位裁剪或范围中止，因为训练环境允许目标角越界，并通过最终电机力矩限幅产生起跳力。部署层也不再检查实测关节位置；宇树电机/固件自身的保护是否介入由机器人系统决定。真机仍检查 actor 输出和关节速度；异常时进入阻尼停机。

## 已有保护

- `LowState` 超时 100 ms：阻尼停机。
- ONNX 输出出现 NaN/Inf 或异常大动作：阻尼停机。
- 实测关节速度异常：阻尼停机。
- 电池电压连续低于 24 V 达 0.25 秒：500 Hz 发布线程切换阻尼；单次起跳
  瞬时压降只告警，不立即停止。
- 任一电机 `lost` 相对启动值增加超过 10：立即切换阻尼。
- 超过 3 个电机温度字段同时变成 0：视为电机状态失效并立即切换阻尼。
- 控制循环连续超时：阻尼停机。
- A 触发前检查直立、低关节速度和足端接触。
- 停机命令持续发送 1 秒，不是只发送一帧。

部署层不再因机身倾角自动停机；RL 在相位结束后仍持续接管恢复。触发新一轮空翻前仍要求机器人直立，`SELECT` 和通信看门狗始终生效。

## 首次真机测试

后空翻具有很高的设备和人身风险。第一次测试应保证：

1. 使用顶部安全绳或保护架，绳索不妨碍俯仰旋转。
2. 人员远离前后运动平面，清空足够大的落地区域。
3. 一人只负责手柄，手指始终放在 `SELECT`。
4. 先运行 `--check`，再只完成站姿和 START 武装，不按 A，确认 60 维 actor 能稳定等待。
5. 确认日志没有控制周期超时、关节异常或足端压力基线异常后，再触发一次 A。
