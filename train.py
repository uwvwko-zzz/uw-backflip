#!/usr/bin/env python3
"""Command-line entry point for Go2 or custom-dog backflip training.

Example:
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --task=go2_backflip \
        --num_envs=4096 \
        --seed=1 \
        --algo=PPO \
        --priv_info \
        --max_iterations=6000 \
        --output_name=go2_backflip_run \
        --headless

The actor always receives the 60-D deployable observation.  The asymmetric
critic always receives the 165-D privileged observation; --priv_info is kept as
an accepted, descriptive compatibility flag for the former command format.
"""

import os

import isaacgym  # noqa: F401

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.dog import DogBackflipCfg, DogBackflipCfgPPO
from legged_gym.envs.go2_backflip.train_asymmetric import train as train_go2
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from rl.Backflip import BackflipRunner


def train_dog(args):
    if args.algo.lower() not in {"ppo", "backflip"}:
        raise ValueError("The dog task supports --algo=PPO only.")

    # dog and dog_backflip are both globally registered aliases.
    args.task = "dog"
    env_cfg = DogBackflipCfg()
    train_cfg = DogBackflipCfgPPO()
    if args.seed is not None:
        train_cfg.seed = args.seed
    env_cfg.seed = train_cfg.seed
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations

    env, _ = task_registry.make_env("dog", args=args, env_cfg=env_cfg)
    log_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "outputs", args.output_name
    )
    runner = BackflipRunner(
        env, class_to_dict(train_cfg), log_dir, device=args.rl_device
    )
    if args.resume:
        if not args.checkpoint_model:
            raise ValueError(
                "--resume requires --checkpoint_model=/path/model.pt"
            )
        runner.load(args.checkpoint_model, load_optimizer=True)
    runner.learn(
        train_cfg.runner.max_iterations, init_at_random_ep_len=True
    )


def main():
    args = get_args()
    if args.task in {"dog", "dog_backflip"}:
        train_dog(args)
    else:
        train_go2(args)


if __name__ == "__main__":
    main()
