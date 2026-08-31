# Go2 Backflip：Isaac Gym 到真机部署

本仓库基于 **Genesis-backflip** ，将其迁移到 Isaac Gym，并完成 Unitree Go2 的 Sim2Real 部署。
当前策略已经完成真机后空翻测试，覆盖以下完整流程：

```text
Isaac Gym 从零训练 -> Isaac Gym 交互验证 -> ONNX 导出
                    -> MuJoCo sim2sim -> Unitree Go2 真机部署
```

策略从起跳、旋转、落地、恢复到等待下一次触发全程由 RL 接管。触发按键只
重启策略相位，不会在一次动作结束后重置机器人状态。

> 后空翻可能造成人身伤害和设备损坏。真机测试必须使用安全绳或保护架，
> 清空前后运动区域，并安排专人负责急停。

## 真机演示

Unitree Go2 真机后空翻，actor 50 Hz 推理、LowCmd 500 Hz 发力，全程由策略接管：

https://github.com/user-attachments/assets/dfa6c1a1-600a-4a7d-8b5f-218bc82b891a

## 主要特性

- Go2 12 自由度后空翻策略，控制频率 50 Hz。
- Actor 使用可在真机获得的 60 维观测，输出 12 维动作。
- Critic 使用 165 维仿真特权观测，仅在训练时使用。
- 两秒多频率相位特征控制后空翻时序。
- 动作延迟、观测延迟、观测噪声、电机能力、质量/惯量和接触随机化。
- Go2HV 20.2/23.4 Nm、13.5--30 rad/s 扭矩-速度包络。
- 头部接触、关节限位、落地冲击和恢复站姿奖励课程。
- Isaac Gym、MuJoCo 和真机共享观测排列、动作缩放、PD 参数和相位定义。
- 真机 actor 50 Hz，最新 LowCmd 以 500 Hz 发布。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `train.py` | Go2 后空翻统一训练入口 |
| `legged_gym/envs/go2_backflip/` | 环境、奖励、随机化和 PPO 配置 |
| `legged_gym/scripts/go2/play.py` | Isaac Gym 交互验证 |
| `rl/Backflip/` | Asymmetric Actor-Critic、PPO、存储和 runner |
| `resources/robots/go2/` | Isaac Gym 使用的 Go2 URDF 和网格 |
| `mujoco/script/pt2onnx.py` | 从 checkpoint 导出 60 维 actor ONNX |
| `mujoco/go2/` | 对齐 Isaac Gym 的 Go2 MJCF 与 sim2sim 脚本 |
| `deploy_real/` | Unitree SDK2 真机部署和配置 |
| `outputs/` | 本地训练结果，已被 `.gitignore` 忽略 |
| `models/` | 可选择提交的最终部署模型 |

## 从零训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task=go2_backflip \
  --algo=PPO \
  --num_envs=4096 \
  --seed=1 \
  --priv_info \
  --max_iterations=6000 \
  --output_name=go2_backflip_run \
  --headless
```

`--priv_info` 是旧命令格式的兼容参数；当前 asymmetric critic 始终使用
165 维特权观测，actor 始终只使用 60 维可部署观测。


## Isaac Gym 验证

通过 `--model=...` 指定训练 checkpoint：

```bash
python legged_gym/scripts/go2/play.py \
  --model=outputs/go2_backflip_run/stage1_nn/model_5000.pt
```

按键：

- `Space`：触发一次后空翻。
- `P`：显式重置环境。
- `Esc`：退出。

默认使用确定性 actor、标称动力学和固定一拍 20 ms 动作延迟。可选参数：

```bash
--sampled       # 从动作分布采样
--randomized    # 使用训练时的动力学和延迟随机化
--num_envs=1
--seed=1
```

后空翻相位结束不会自动重置物理状态，actor 会继续负责落地、恢复和等待。

## 策略接口

Actor 观测严格按以下顺序拼接：

| 观测 | 维度 | 缩放 |
| --- | ---: | ---: |
| 机身角速度 | 3 | 0.25 |
| 机身坐标系重力方向 | 3 | 1.0 |
| `q - default_q` | 12 | 1.0 |
| 关节速度 | 12 | 0.05 |
| 当前动作 | 12 | 1.0 |
| 上一拍动作 | 12 | 1.0 |
| 相位特征 | 6 | 1.0 |
| 合计 | 60 | |

动作输出为 12 维，关节顺序为：

```text
FL(hip, thigh, calf), FR(hip, thigh, calf),
RL(hip, thigh, calf), RR(hip, thigh, calf)
```

主要控制参数：

```text
policy_dt       0.02 s (50 Hz)
phase_duration  2.0 s
action_scale    0.5
Kp / Kd         40 / 1
```

相位使用 `sin/cos(phi)`、`sin/cos(phi/2)` 和 `sin/cos(phi/4)`，其中：

```text
phi = pi * phase_time / 2
```

## Asymmetric Critic：165 维特权观测

Actor 只使用真机可获得的 60 维观测；critic 在训练时额外看到仿真器中的
干净状态和每个环境实际采样到的随机化参数：

```text
critic observation = state_privileged(64) + dynamics_privileged(101)
                   = 165
```

64 维状态部分：

| Critic 状态 | 维度 | 说明 |
| --- | ---: | --- |
| 基座高度 | 1 | 世界坐标系 `base_z` |
| 基座线速度 | 3 | 机身坐标系线速度 |
| 干净本体状态 | 30 | 角速度 3、重力 3、关节位置 12、关节速度 12 |
| 当前动作 | 12 | 当前 actor 输出 |
| 上一拍动作 | 12 | 动作历史 |
| 相位特征 | 6 | 与 actor 相同的多频率相位 |
| 合计 | 64 | |

这 30 维本体状态不添加 actor 侧的传感器噪声、零偏和观测延迟。

101 维动力学部分：

| 随机化/动力学参数 | 维度 |
| --- | ---: |
| 12 个电机力矩倍率 | 12 |
| 12 个电机速度倍率 | 12 |
| 动作延迟与观测延迟档位 | 2 |
| 16 个腿部刚体质量倍率 | 16 |
| 16 个腿部刚体惯量倍率 | 16 |
| 摩擦、恢复系数、contact offset | 3 |
| 基座质量与三轴质心偏移 | 4 |
| 12 个实际 Kp 与 12 个实际 Kd | 24 |
| 12 个电机零位偏移 | 12 |
| 合计 | 101 |

这种 asymmetric 设计让 value function 知道某条轨迹为什么更难，例如电机
更弱、延迟更大或地面更滑，从而更准确地估计回报、降低 PPO 优势估计方差。
部署只导出 actor，因此这 101 维仿真信息不会进入 ONNX，也不会造成真机
观测依赖。

## 奖励函数设计

整体思路是“先学会完整后翻，再逐渐学会安全、稳定地后翻”。奖励允许为负，
各非零项按控制周期 `dt` 缩放后求和。

1. **起跳塑形**：起跳前约束默认站姿、机身高度和足端贴地；在
   `0.50--0.75 s` 奖励向上速度，促使机器人产生有效起跳冲量。
2. **旋转塑形**：在 `0.50--1.00 s` 奖励后翻角速度，使用相位目标姿态提供
   连续引导，同时惩罚偏航和侧滚。`rotation_progress` 只奖励以前没有达到过
   的后翻角度，防止策略通过反复摆动刷奖励。
3. **完成事件**：累计后翻达到 `5.50 rad` 给一次 `flip_completion`；达到
   `5.80 rad` 后，还要满足落地直立、足够高度、低俯仰速度、接近默认姿态
   和至少三足接触，才给一次 `flip_success`。两个事件配置权重均为 `500`，
   使完整旋转并站住明显优于高速半翻或倒地。
4. **落地恢复**：从 `1.40 s` 开始奖励直立、恢复高度、默认关节位置、低
   机身速度和足端接触。相位在 `2.0 s` 冻结，但 episode 延续到 `3.0 s`，
   因此落地和恢复仍由 RL 完成，而不是依赖环境重置。
5. **动作质量和安全**：动作变化率、二阶差分、左右对称、后腿协调、关节
   速度/位置、actor 原始目标越界、头部/躯干接触和落地冲击分别受到惩罚。

安全与部分动作质量项采用课程：初始只有完整权重的 `5%`，前 6000 个控制步
保持较轻，之后在 30000 步内线性增加到 `100%`。这是为了让策略先发现可行
后空翻，再消除头部碰地、关节越界、后腿乱动和重落地；如果从第 0 步就施加
全部强惩罚，策略很容易学成“完全不跳”。

## 动态力矩剪裁

策略动作先变成位置目标，仿真每个 5 ms 物理步计算原始 PD 力矩：

```text
q_target = default_q + 0.5 * slew_limited_action
tau_pd   = Kp * (q_target - q) - Kd * dq
```

目标位置本身也按 `13.5 rad/s` 限制每步变化量，避免 actor 用瞬时目标跳变
制造非真实冲量。随后并不是简单固定裁剪到 `[-20.2, 20.2] Nm`，而是根据
当前关节速度计算电机可用能力：

```text
alpha = 1                               , |dq| < x1
alpha = (x2 - |dq|) / (x2 - x1)        , x1 <= |dq| < x2
alpha = 0                               , |dq| >= x2

x1 = 13.5 * motor_velocity_scale
x2 = 30.0 * motor_velocity_scale
```

力矩与速度同向表示电机继续驱动加速，反向表示制动：

```text
tau_peak = 20.2 Nm,  dq * tau_pd > 0   # 驱动
tau_peak = 23.4 Nm,  dq * tau_pd <= 0  # 制动

tau_limit = alpha * torque_scale * tau_peak
tau_cmd   = clamp(tau_pd, -tau_limit, +tau_limit)
```

训练中每个电机的 `torque_scale` 随机为 `0.80--1.00`，速度倍率随机为
`0.85--1.00`。这表达了电机反电动势的基本特性：低速可用峰值力矩，超过
13.5 rad/s 后能力线性下降，到 30 rad/s 时降为零；制动力又可以略高于
驱动力。代码不会硬改关节速度，而是缩小真实可用力矩，让高速策略自然暴露
问题。Isaac Gym、MuJoCo 和真机 500 Hz 直力矩层使用相同思想，避免策略在
仿真中依靠真机无法提供的高速大力矩完成动作。

## 导出 ONNX

将完整 `.pt` checkpoint 中的 actor 导出为单个 ONNX：

```bash
python mujoco/script/pt2onnx.py \
  --model=outputs/go2_backflip_run/stage1_nn/model_5000.pt \
  --out=outputs/go2_backflip_run/model_5000_actor.onnx
```

导出脚本会检查 actor 输入必须为 60 维、输出必须为 12 维。ONNX 只包含
actor，不包含训练时的 critic。

## MuJoCo sim2sim

运行交互验证：

```bash
python mujoco/go2/play_onnx.py \
  --onnx=outputs/go2_backflip_run/model_5000_actor.onnx
```

MuJoCo 按键：

- `Space`：触发后空翻，只重启策略相位。
- `R`：显式物理重置。
- `Esc`：退出。
