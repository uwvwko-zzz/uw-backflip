# Go2 后空翻 MuJoCo 验证

`go2_isaac.xml` 由 Isaac Gym 训练使用的
`resources/robots/go2/urdf/go2_description.urdf` 转换生成。验证脚本默认使用
该模型，不再默认使用 `mujoco_menagerie` 中经过简化和软脚底处理的模型。

## 依赖

```bash
python -m pip install mujoco onnxruntime pynput
```

## 重新生成 MJCF

```bash
cd ~/桌面/uw-backflip
python mujoco/script/urdf_to_mjcf.py
```

转换器会保留 URDF 的刚体层级、质量、惯量和碰撞体，并应用训练中的关键
Isaac 资产选项：自由基座、`density=0.001`、圆柱碰撞体转胶囊体，以及
12 个受 URDF 力矩限幅约束的执行器。模型不包含 Menagerie 的脚底软接触。
固定关节按照 Isaac 的实际加载结果折叠为 19 个机器人刚体，并保留独立的
4 个 foot 和 2 个 Head 刚体。地面接触参数单独设置，不影响机器人自碰撞。
首次转换会把 Menagerie 中同源的 OBJ 复制到 `mujoco/go2/assets/`，这些
网格只负责渲染，不参与质量、碰撞或接触计算。

## 运行

```bash
cd ~/桌面/uw-backflip
python mujoco/go2/play_onnx.py \
  --onnx outputs/backflip_v4/stage1_nn/model_5000_actor.onnx \
  --start
```

默认模型是 `mujoco/go2/go2_isaac.xml`。可使用 `--mjcf` 显式选择其他
MJCF。按键如下：

- `Space`：触发一次后空翻，只重启相位，不重置机器人
- `R`：物理重置机器人
- `Esc`：退出

无界面快速验证：

```bash
python mujoco/go2/play_onnx.py \
  --onnx outputs/backflip_v4/stage1_nn/model_5000_actor.onnx \
  --start --headless --no-realtime --duration 5
```
