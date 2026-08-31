#!/usr/bin/env python3
"""Visualize a trained Go2 backflip policy in Isaac Gym.

Example:
    python legged_gym/scripts/go2/play.py \
        --model=outputs/go2_backflip_run/stage1_nn/model_5000.pt

The policy waits at its initial phase. Press SPACE to execute one backflip,
P to reset, and ESC (or close the viewer) to quit.
"""

import argparse
import os
import sys
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401
import torch

from legged_gym.envs.go2_backflip import (
    Go2Backflip,
    Go2BackflipCfg,
    Go2BackflipCfgPPO,
)
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict, get_args
from rl.Backflip import AsymmetricActorCritic


def _parse_play_args():
    parser = argparse.ArgumentParser(
        description="Visualize an asymmetric Go2 backflip checkpoint."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="CHECKPOINT",
        help="Backflip .pt checkpoint, for example --model=outputs/run/stage1_nn/model_5000.pt",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Sample actions; by default the deterministic actor mean is used.",
    )
    parser.add_argument(
        "--randomized",
        action="store_true",
        help="keep the training domain randomization; default is nominal deterministic validation",
    )
    play_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return play_args


class Keyboard:
    """Optional pynput keyboard listener used for SPACE/P/ESC controls."""

    def __init__(self):
        self.running = True
        self._trigger_requested = False
        self._reset_requested = False
        self._space_down = False
        self._lock = threading.Lock()
        self.listener = None

    def start(self):
        try:
            from pynput import keyboard
        except ImportError:
            print("pynput is unavailable; use the viewer ESC key to quit.")
            return

        def on_press(key):
            if key == keyboard.Key.esc:
                self.running = False
                return
            if key == keyboard.Key.space:
                with self._lock:
                    if not self._space_down:
                        self._trigger_requested = True
                    self._space_down = True
                return
            try:
                if key.char and key.char.lower() == "p":
                    with self._lock:
                        self._reset_requested = True
            except AttributeError:
                pass

        def on_release(key):
            if key == keyboard.Key.space:
                with self._lock:
                    self._space_down = False

        self.listener = keyboard.Listener(
            on_press=on_press, on_release=on_release
        )
        self.listener.start()

    def consume_trigger(self):
        with self._lock:
            requested = self._trigger_requested
            self._trigger_requested = False
        return requested

    def consume_reset(self):
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
        return requested

    def stop(self):
        if self.listener is not None:
            self.listener.stop()


def _load_checkpoint(actor_critic, model_path, device):
    model_path = os.path.abspath(
        os.path.expandvars(os.path.expanduser(model_path))
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "actor_state_dict" in checkpoint:
        # Kept only to produce a useful shape/key error for older checkpoints.
        state_dict = checkpoint["actor_state_dict"]
    else:
        state_dict = checkpoint

    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()
    return model_path, checkpoint.get("iteration", checkpoint.get("iter", "unknown"))


def main():
    play_args = _parse_play_args()
    sim_args = get_args()
    sim_args.task = "go2_backflip"
    sim_args.headless = False
    sim_args.num_envs = play_args.num_envs
    sim_args.seed = play_args.seed

    env_cfg = Go2BackflipCfg()
    train_cfg = Go2BackflipCfgPPO()
    env_cfg.seed = play_args.seed
    env_cfg.env.num_envs = play_args.num_envs
    env_cfg.env.num_privileged_obs = 165
    env_cfg.env.num_env_priv_obs = 0
    if not play_args.randomized:
        # Nominal sim2sim validation must be reproducible. In particular, the
        # training environment samples 0/20/40-ms actuator delay per episode;
        # validation fixes it to the real-deployment nominal value of 20 ms.
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_center = False
        env_cfg.domain_rand.randomize_Motor_Offset = False
        env_cfg.domain_rand.randomize_kp_scale = False
        env_cfg.domain_rand.randomize_kd_scale = False
        env_cfg.domain_rand.randomize_motor_strength = False
        env_cfg.domain_rand.randomize_joint_friction = False
        env_cfg.domain_rand.randomize_torque_scale = False
        env_cfg.domain_rand.randomize_motor_velocity = False
        env_cfg.domain_rand.randomize_limb_mass = False
        env_cfg.domain_rand.randomize_limb_inertia = False
        env_cfg.domain_rand.randomize_contact = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.control.max_delay_steps = 1
        env_cfg.control.max_observation_delay_steps = 0

    task_registry.register("go2_backflip", Go2Backflip, env_cfg, train_cfg)
    env, _ = task_registry.make_env("go2_backflip", args=sim_args, env_cfg=env_cfg)
    if not play_args.randomized:
        # max_delay_steps controls buffer allocation, while env_delay_steps is
        # the actual per-environment selection. Force buffer index 1 = 20 ms.
        env.env_delay_steps.fill_(1)

    actor_critic = AsymmetricActorCritic(
        env.num_obs,
        env.num_privileged_obs,
        env.num_actions,
        **class_to_dict(train_cfg.policy),
    ).to(sim_args.rl_device)
    model_path, iteration = _load_checkpoint(
        actor_critic, play_args.model, sim_args.rl_device
    )

    obs_dict = env.reset()
    if not play_args.randomized:
        env.env_delay_steps.fill_(1)

    # Evaluation is one continuous physical trajectory. Phase completion must
    # not teleport the robot back to its initial root/DOF state.
    def disable_automatic_reset():
        env.time_out_buf.zero_()
        env.reset_buf.zero_()

    env.check_termination = disable_automatic_reset
    keyboard = Keyboard()
    keyboard.start()
    step = 0
    backflip_active = False
    hold_phase = 0

    print("=" * 72)
    print(f"Backflip model : {model_path}")
    print(f"Iteration      : {iteration}")
    print(f"Environments   : {env.num_envs}")
    print(f"Policy mode    : {'sampled' if play_args.sampled else 'deterministic'}")
    print(
        "Physics mode   : "
        f"{'training randomization' if play_args.randomized else 'nominal deterministic (20-ms delay)'}"
    )
    print(f"Episode period : {env.max_episode_length * env.dt:.2f}s")
    print("Controls       : SPACE one backflip | P reset | ESC/close viewer quit")
    print("=" * 72)

    try:
        while keyboard.running:
            if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
                break
            if keyboard.consume_reset():
                obs_dict = env.reset()
                if not play_args.randomized:
                    env.env_delay_steps.fill_(1)
                backflip_active = False
                hold_phase = 0
                print("[P] phase and robot reset")
            if keyboard.consume_trigger() and not backflip_active:
                env.episode_length_buf[:] = 0
                env.compute_observations()
                obs_dict = env.get_observations()
                backflip_active = True
                print("[SPACE] backflip triggered")

            with torch.inference_mode():
                actor_obs = obs_dict["obs"].to(sim_args.rl_device)
                if play_args.sampled:
                    actions = actor_critic.act(actor_obs)
                else:
                    actions = actor_critic.act_inference(actor_obs)
                obs_dict, rewards, _, _ = env.step(actions.detach())

            if backflip_active:
                if env.episode_length_buf[0] >= env.max_episode_length:
                    backflip_active = False
                    hold_phase = int(env.max_episode_length)
                    env.episode_length_buf[:] = hold_phase
                    env.compute_observations()
                    obs_dict = env.get_observations()
                    print(
                        "[BACKFLIP] cycle complete; actor recovering/holding "
                        "the default pose"
                    )
            else:
                # Freeze only the policy phase. Physics and actor control remain
                # continuous; no robot state is reset or overwritten.
                env.episode_length_buf[:] = hold_phase
                env.compute_observations()
                obs_dict = env.get_observations()

            if step % 10 == 0:
                phase_time = env.episode_length_buf[0].item() * env.dt
                base_z = env.root_states[0, 2].item()
                pitch_rate = env.base_ang_vel[0, 1].item()
                print(
                    f"step {step:5d} | {'FLIP' if backflip_active else 'WAIT':4s} | "
                    f"phase {phase_time:4.2f}s | "
                    f"base_z {base_z:5.3f}m | pitch_rate {pitch_rate:+6.3f}rad/s | "
                    f"reward {rewards[0].item():+7.3f}",
                    flush=True,
                )
            step += 1
    finally:
        keyboard.stop()

    print("Done.")


if __name__ == "__main__":
    main()
