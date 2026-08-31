#!/usr/bin/env python3
"""Export the deterministic Go2 backflip actor from a training checkpoint.

The exported ONNX contains only the actor used on the robot:

    observation (60) -> actor MLP -> action (12)

The critic, exploration ``std`` and optimizer state are intentionally omitted.
"""

import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn


EXPECTED_OBSERVATION_DIM = 60
EXPECTED_ACTION_DIM = 12


def resolve_model_path(positional_path, option_path):
    model_path = option_path or positional_path
    if not model_path:
        raise ValueError("请通过位置参数、--model 或 --checkpoint 指定模型文件")

    model_path = os.path.abspath(
        os.path.expandvars(os.path.expanduser(model_path))
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return model_path


def resolve_output_path(output_arg, model_path):
    if output_arg:
        output_path = os.path.abspath(
            os.path.expandvars(os.path.expanduser(output_arg))
        )
        if not output_path.lower().endswith(".onnx"):
            output_path = os.path.join(output_path, "backflip_actor.onnx")
    else:
        model_stem = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(
            os.path.dirname(model_path), f"{model_stem}_actor.onnx"
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_path


def extract_model_state_dict(checkpoint):
    """Accept current runner checkpoints and common older checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 必须是字典")

    for key in ("model_state_dict", "actor_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    else:
        # Also accept a raw torch state_dict.
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise KeyError(
                "checkpoint 中未找到 model_state_dict/actor_state_dict/state_dict"
            )

    # DataParallel checkpoints add this prefix.
    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def extract_actor_state_dict(state_dict):
    actor_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("actor."):
            actor_state[key[len("actor."):]] = value

    # A checkpoint saved directly from model.actor has keys such as 0.weight.
    if not actor_state:
        actor_state = OrderedDict(
            (key, value) for key, value in state_dict.items()
            if key.split(".", 1)[0].isdigit()
        )

    if not actor_state:
        raise KeyError("checkpoint 中没有找到 actor 网络权重")
    return actor_state


def build_actor(actor_state):
    linear_indices = sorted(
        int(key.split(".", 1)[0])
        for key in actor_state
        if key.endswith(".weight") and key.split(".", 1)[0].isdigit()
    )
    if not linear_indices:
        raise ValueError("无法从 actor 权重推断线性层")

    layers = []
    previous_out_dim = None
    for layer_number, state_index in enumerate(linear_indices):
        weight_key = f"{state_index}.weight"
        bias_key = f"{state_index}.bias"
        if bias_key not in actor_state:
            raise KeyError(f"actor 缺少参数: {bias_key}")

        weight = actor_state[weight_key]
        if weight.ndim != 2:
            raise ValueError(f"{weight_key} 不是二维线性层权重")
        out_dim, in_dim = weight.shape
        if previous_out_dim is not None and in_dim != previous_out_dim:
            raise ValueError(
                f"actor 层尺寸不连续: 上一层输出 {previous_out_dim}，"
                f"当前层输入 {in_dim}"
            )

        layers.append(nn.Linear(in_dim, out_dim))
        if layer_number < len(linear_indices) - 1:
            layers.append(nn.ELU())
        previous_out_dim = out_dim

    actor = nn.Sequential(*layers)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    return actor, layers[0].in_features, layers[-1].out_features


def main():
    parser = argparse.ArgumentParser(
        description="将 Go2 后空翻 checkpoint 中的 actor 导出为 ONNX"
    )
    parser.add_argument("model_path", nargs="?", help=".pt/.pth 模型路径")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", dest="model_option", help=".pt/.pth 模型路径")
    model_group.add_argument(
        "--checkpoint", dest="model_option", help="--model 的兼容名称"
    )
    parser.add_argument(
        "--out", default=None,
        help="输出 .onnx 文件或目录；默认生成 <模型名>_actor.onnx",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset，默认 11")
    args = parser.parse_args()

    try:
        model_path = resolve_model_path(args.model_path, args.model_option)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    output_path = resolve_output_path(args.out, model_path)
    print(f"loading:  {model_path}")

    # Keep compatibility with the project's PyTorch 2.0 environment.
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = extract_model_state_dict(checkpoint)
    actor_state = extract_actor_state_dict(state_dict)
    actor, observation_dim, action_dim = build_actor(actor_state)

    if observation_dim != EXPECTED_OBSERVATION_DIM:
        raise ValueError(
            f"该模型 actor 输入为 {observation_dim} 维，不是当前后空翻策略的 "
            f"{EXPECTED_OBSERVATION_DIM} 维"
        )
    if action_dim != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"该模型输出为 {action_dim} 维，不是 Go2 的 "
            f"{EXPECTED_ACTION_DIM} 维动作"
        )

    print(f"actor:    {actor}")
    print(f"input:    observation [batch, {observation_dim}]")
    print(f"output:   action      [batch, {action_dim}]")
    print("layout:   ang_vel(3), gravity(3), dof_pos(12), dof_vel(12),")
    print("          action(12), last_action(12), phase(6)")

    dummy_observation = torch.zeros(1, observation_dim, dtype=torch.float32)
    with torch.no_grad():
        actor(dummy_observation)

    torch.onnx.export(
        actor,
        dummy_observation,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
        },
    )
    print(f"exported: {output_path}")


if __name__ == "__main__":
    main()
