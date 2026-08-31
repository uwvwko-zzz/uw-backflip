"""Train the randomized Go2 backflip task from scratch or an explicit resume."""

import os
import isaacgym  # noqa: F401

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2_backflip import Go2Backflip, Go2BackflipCfg, Go2BackflipCfgPPO
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from rl.Backflip import BackflipRunner


SUPPORTED_TASKS = {"go2_backflip", "go2-backflip", "backflip"}
SUPPORTED_ALGORITHMS = {"ppo", "backflip"}


def train(args):
    """Build the configured environment and run asymmetric backflip PPO."""

    # get_args() comes from the original multi-task project and still carries
    # its old default task name.  Treat that default as an omitted --task.
    if args.task == "anymal_c_flat":
        args.task = "go2_backflip"
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task {args.task!r}. This repository now contains only "
            "--task=go2_backflip."
        )
    if args.algo.lower() not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm {args.algo!r}. GenHis and the other legacy "
            "runners were removed; use --algo=PPO."
        )

    args.task = "go2_backflip"
    env_cfg = Go2BackflipCfg()
    env_cfg.env.num_privileged_obs = 165
    # This task builds its complete asymmetric critic observation directly.
    # The legacy RMA-only auxiliary privileged buffer must stay disabled.
    env_cfg.env.num_env_priv_obs = 0
    train_cfg = Go2BackflipCfgPPO()
    if args.seed is not None:
        train_cfg.seed = args.seed
    env_cfg.seed = train_cfg.seed
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    task_registry.register("go2_backflip", Go2Backflip, env_cfg, train_cfg)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "outputs", args.output_name)
    runner = BackflipRunner(env, class_to_dict(train_cfg), log_dir, device=args.rl_device)
    if args.resume:
        # Resume only checkpoints produced by this 60-D actor / 165-D critic task.
        if not args.checkpoint_model:
            raise ValueError("--resume requires --checkpoint_model=/path/to/model.pt")
        runner.load(args.checkpoint_model, load_optimizer=True)
    # Preserve the phase-distributed scratch-training setup used by v2.
    runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=True)


def main():
    train(get_args())


if __name__ == "__main__":
    main()
