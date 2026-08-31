#!/usr/bin/env python3
"""Interactive Isaac Gym validation for a trained dog checkpoint."""

import argparse
import os
import sys
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.dog import DogBackflip, DogBackflipCfg, DogBackflipCfgPPO
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict, get_args
from rl.Backflip import AsymmetricActorCritic


class Keyboard:
    """Optional SPACE/P/ESC keyboard control."""

    def __init__(self):
        self.running = True
        self._trigger = False
        self._reset = False
        self._space_down = False
        self._lock = threading.Lock()
        self.listener = None

    def start(self):
        try:
            from pynput import keyboard
        except ImportError:
            print("pynput is unavailable; close the viewer to quit.")
            return

        def on_press(key):
            if key == keyboard.Key.esc:
                self.running = False
            elif key == keyboard.Key.space:
                with self._lock:
                    if not self._space_down:
                        self._trigger = True
                    self._space_down = True
            else:
                try:
                    if key.char and key.char.lower() == "p":
                        with self._lock:
                            self._reset = True
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
            requested = self._trigger
            self._trigger = False
            return requested

    def consume_reset(self):
        with self._lock:
            requested = self._reset
            self._reset = False
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
    state_dict = checkpoint.get(
        "model_state_dict", checkpoint.get("actor_state_dict", checkpoint)
    )
    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()
    iteration = checkpoint.get("iteration", checkpoint.get("iter", "unknown"))
    return model_path, iteration


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a dog backflip checkpoint."
    )
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--randomized", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _make_nominal(cfg):
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_center = False
    cfg.domain_rand.randomize_Motor_Offset = False
    cfg.domain_rand.randomize_kp_scale = False
    cfg.domain_rand.randomize_kd_scale = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_joint_friction = False
    cfg.domain_rand.randomize_torque_scale = False
    cfg.domain_rand.randomize_motor_velocity = False
    cfg.domain_rand.randomize_limb_mass = False
    cfg.domain_rand.randomize_limb_inertia = False
    cfg.domain_rand.randomize_contact = False
    cfg.domain_rand.push_robots = False
    cfg.control.max_delay_steps = 1
    cfg.control.max_observation_delay_steps = 0


def main():
    play_args = _parse_args()
    sim_args = get_args()
    sim_args.task = "dog_backflip"
    sim_args.headless = False
    sim_args.num_envs = play_args.num_envs
    sim_args.seed = play_args.seed

    env_cfg = DogBackflipCfg()
    train_cfg = DogBackflipCfgPPO()
    env_cfg.seed = play_args.seed
    env_cfg.env.num_envs = play_args.num_envs
    if not play_args.randomized:
        _make_nominal(env_cfg)

    env, _ = task_registry.make_env(
        "dog_backflip", args=sim_args, env_cfg=env_cfg
    )
    if not play_args.randomized:
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

    def disable_automatic_reset():
        env.time_out_buf.zero_()
        env.reset_buf.zero_()

    env.check_termination = disable_automatic_reset
    keyboard = Keyboard()
    keyboard.start()
    step = 0
    active = False
    hold_phase = 0

    print("=" * 72)
    print(f"Dog model      : {model_path}")
    print(f"Iteration      : {iteration}")
    print("Controls       : SPACE one backflip | P reset | ESC quit")
    print(
        "Physics mode   : "
        + ("randomized" if play_args.randomized else "nominal, 20-ms delay")
    )
    print("=" * 72)
    try:
        while keyboard.running:
            if env.viewer is not None and env.gym.query_viewer_has_closed(
                env.viewer
            ):
                break
            if keyboard.consume_reset():
                obs_dict = env.reset()
                if not play_args.randomized:
                    env.env_delay_steps.fill_(1)
                active = False
                hold_phase = 0
                print("[P] robot and phase reset")
            if keyboard.consume_trigger() and not active:
                env.episode_length_buf.zero_()
                env.compute_observations()
                obs_dict = env.get_observations()
                active = True
                print("[SPACE] dog backflip triggered")

            with torch.inference_mode():
                actor_obs = obs_dict["obs"].to(sim_args.rl_device)
                actions = (
                    actor_critic.act(actor_obs)
                    if play_args.sampled
                    else actor_critic.act_inference(actor_obs)
                )
                obs_dict, rewards, _, _ = env.step(actions.detach())

            if active and env.episode_length_buf[0] >= env.max_episode_length:
                active = False
                hold_phase = int(env.max_episode_length)
                print("[BACKFLIP] cycle complete; RL continues holding")
            if not active:
                env.episode_length_buf[:] = hold_phase
                env.compute_observations()
                obs_dict = env.get_observations()

            if step % 10 == 0:
                phase = env.episode_length_buf[0].item() * env.dt
                print(
                    f"step {step:5d} | "
                    f"{'FLIP' if active else 'WAIT':4s} | "
                    f"phase {phase:4.2f}s | "
                    f"base_z {env.root_states[0, 2].item():5.3f}m | "
                    f"pitch_rate {env.base_ang_vel[0, 1].item():+6.3f} | "
                    f"reward {rewards[0].item():+7.3f}",
                    flush=True,
                )
            step += 1
    finally:
        keyboard.stop()


if __name__ == "__main__":
    main()
