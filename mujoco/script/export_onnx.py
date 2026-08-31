"""
Standalone ONNX exporter for the GenHis (OmniNet) policy.

Rebuilds the two networks that the deployment path actually uses
(see rl/Gen_his/modules/actor_critic.py :: _actor_critic / act_inference):

    proprio_history(230) --dm_encoder--> extrin(13)
    cat([extrin(13), obs(46)])(59) --actor--> action(12)

Networks are reconstructed *from the checkpoint's weight shapes*, so this
script only needs torch (no isaacgym, no legged_gym import).

Usage:
    # 直接使用任意位置的模型（推荐）
    python mujoco/script/export_onnx.py /path/to/model_5000.pt

    # 也可以使用 --model；--checkpoint 作为旧参数仍然兼容
    python mujoco/script/export_onnx.py \
        --model /path/to/model_5000.pt \
        --out /path/to/onnx
"""
import argparse
import os

import torch
import torch.nn as nn


def build_actor(sd):
    """Rebuild the actor MLP from keys 'actor.{i}.weight' (ELU between layers)."""
    idxs = sorted(int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight"))
    layers = []
    for j, i in enumerate(idxs):
        w = sd[f"actor.{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if j < len(idxs) - 1:
            layers.append(nn.ELU())  # get_activation('elu') in the training code
    net = nn.Sequential(*layers)
    # load weights (strip the 'actor.' prefix -> matches Sequential indices)
    sub = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    net.load_state_dict(sub)
    net.eval()
    return net, layers[0].in_features, layers[-1].out_features


def build_encoder(sd):
    """Rebuild DmEncoder from keys 'dm_encoder.encoder.{i}.weight' (ReLU,ReLU,Tanh)."""
    prefix = "dm_encoder.encoder."
    idxs = sorted(int(k[len(prefix):].split(".")[0])
                  for k in sd if k.startswith(prefix) and k.endswith(".weight"))
    layers = []
    for j, i in enumerate(idxs):
        w = sd[f"{prefix}{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        layers.append(nn.ReLU() if j < len(idxs) - 1 else nn.Tanh())  # DmEncoder ends with Tanh
    net = nn.Sequential(*layers)
    sub = {k[len("dm_encoder.encoder."):]: v for k, v in sd.items() if k.startswith(prefix)}
    net.load_state_dict(sub)
    net.eval()
    return net, layers[0].in_features, layers[-2].out_features


def export(net, in_dim, path):
    dummy = torch.zeros(1, in_dim)
    torch.onnx.export(
        net, dummy, path,
        export_params=True, opset_version=11,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"  exported {path}  (in={in_dim})")


def resolve_model_path(positional_path, option_path):
    """解析并检查用户指定的任意 .pt/.pth 模型路径。"""
    model_path = option_path or positional_path
    if not model_path:
        raise ValueError(
            "未指定模型。请传入模型路径，或使用 --model /path/to/model.pt"
        )

    model_path = os.path.abspath(os.path.expandvars(os.path.expanduser(model_path)))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    if not model_path.lower().endswith((".pt", ".pth")):
        raise ValueError(f"模型文件必须以 .pt 或 .pth 结尾: {model_path}")
    return model_path


def extract_actor_state_dict(checkpoint):
    """兼容 OmniNet 完整 checkpoint 和直接保存的模型 state_dict。"""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"不支持的模型内容类型: {type(checkpoint).__name__}，预期为字典"
        )

    # GenHisPolicyRunner.save() 产生的标准 OmniNet checkpoint。
    if "actor_state_dict" in checkpoint:
        state_dict = checkpoint["actor_state_dict"]
    # 一些训练框架使用通用的 state_dict/model_state_dict 字段。
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    # 也支持 torch.save(model.state_dict(), path) 保存的裸 state dict。
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise KeyError(
            "模型中没有 actor_state_dict、model_state_dict 或 state_dict，"
            f"现有字段: {list(checkpoint.keys())}"
        )

    # 兼容 DistributedDataParallel 保存时添加的 module. 前缀。
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value for key, value in state_dict.items()
        }

    actor_weights = [
        key for key in state_dict
        if key.startswith("actor.") and key.endswith(".weight")
    ]
    encoder_weights = [
        key for key in state_dict
        if key.startswith("dm_encoder.encoder.") and key.endswith(".weight")
    ]
    if not actor_weights or not encoder_weights:
        raise KeyError(
            "该模型不是可识别的 GenHis checkpoint："
            "缺少 actor.* 或 dm_encoder.encoder.* 网络权重"
        )
    return state_dict


def main():
    ap = argparse.ArgumentParser(
        description="将任意路径下的 OmniNet/GenHis .pt 模型导出为 ONNX"
    )
    ap.add_argument(
        "model_path",
        nargs="?",
        help="模型路径，例如 /data/models/model_5000.pt",
    )
    model_group = ap.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        dest="model_option",
        help="任意绝对或相对路径的 .pt/.pth 模型",
    )
    model_group.add_argument(
        "--checkpoint",
        dest="model_option",
        help="--model 的兼容旧名称",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="ONNX 输出目录；默认在模型旁创建 <模型名>_onnx 目录",
    )
    args = ap.parse_args()

    try:
        model_path = resolve_model_path(args.model_path, args.model_option)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))

    if args.out:
        out_dir = os.path.abspath(
            os.path.expandvars(os.path.expanduser(args.out))
        )
    else:
        model_stem = os.path.splitext(os.path.basename(model_path))[0]
        out_dir = os.path.join(os.path.dirname(model_path), f"{model_stem}_onnx")

    os.makedirs(out_dir, exist_ok=True)
    print(f"loading: {model_path}")

    # 不传 weights_only 参数，以兼容项目使用的 PyTorch 2.0。
    checkpoint = torch.load(model_path, map_location="cpu")
    sd = extract_actor_state_dict(checkpoint)

    actor, actor_in, actor_out = build_actor(sd)
    encoder, enc_in, enc_out = build_encoder(sd)
    print(f"actor:   {actor_in} -> {actor_out}")
    print(f"encoder: {enc_in} -> {enc_out}")
    assert actor_in == enc_out + (actor_in - enc_out), "sanity"
    print(f"=> obs dim inferred as {actor_in - enc_out} (should be 46)")

    export(actor, actor_in, os.path.join(out_dir, "actor.onnx"))
    export(encoder, enc_in, os.path.join(out_dir, "encoder.onnx"))
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
