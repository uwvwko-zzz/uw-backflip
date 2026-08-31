import os
import re
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from .actor_critic import AsymmetricActorCritic
from .ppo import AsymmetricPPO


class BackflipRunner:
    def __init__(self, env, train_cfg, log_dir, device="cpu"):
        self.env = env
        self.device = device
        self.cfg = train_cfg["runner"]
        self.actor_critic = AsymmetricActorCritic(
            env.num_obs,
            env.num_privileged_obs,
            env.num_actions,
            **train_cfg["policy"],
        ).to(device)
        self.alg = AsymmetricPPO(
            self.actor_critic, device=device, **train_cfg["algorithm"]
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [env.num_obs],
            [env.num_privileged_obs],
            [env.num_actions],
        )
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, "stage1_nn")
        os.makedirs(self.model_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(log_dir, "stage1_tb"), flush_secs=10)
        self.current_learning_iteration = 0
        self.total_steps = 0
        self.total_time = 0.0
        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        reward_history = deque(maxlen=100)
        episode_length_history = deque(maxlen=100)
        episode_rewards = torch.zeros(self.env.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.env.num_envs, device=self.device)
        start_iteration = self.current_learning_iteration
        end_iteration = start_iteration + num_learning_iterations

        for iteration in range(start_iteration, end_iteration):
            ep_infos = []
            collection_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    self.alg.process_env_step(rewards, dones, infos)
                    if "episode" in infos:
                        ep_infos.append({
                            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                            for key, value in infos["episode"].items()
                        })
                    episode_rewards += rewards
                    episode_lengths += 1
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if len(done_ids):
                        reward_history.extend(episode_rewards[done_ids].cpu().tolist())
                        episode_length_history.extend(episode_lengths[done_ids].cpu().tolist())
                        episode_rewards[done_ids] = 0.0
                        episode_lengths[done_ids] = 0.0
                collection_time = time.time() - collection_start
                learning_start = time.time()
                self.alg.compute_returns(obs)

            value_loss, surrogate_loss = self.alg.update()
            learning_time = time.time() - learning_start
            iteration_time = collection_time + learning_time
            self.total_time += iteration_time
            self.total_steps += self.num_steps_per_env * self.env.num_envs
            self.current_learning_iteration = iteration + 1
            self.writer.add_scalar("Loss/value", value_loss, iteration)
            self.writer.add_scalar("Loss/surrogate", surrogate_loss, iteration)
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
            self.writer.add_scalar("Policy/noise_std", self.actor_critic.std.mean(), iteration)
            self.writer.add_scalar("Perf/fps", self.num_steps_per_env * self.env.num_envs / iteration_time, iteration)
            self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
            self.writer.add_scalar("Perf/learning_time", learning_time, iteration)
            if reward_history:
                self.writer.add_scalar(
                    "Train/mean_episode_reward",
                    sum(reward_history) / len(reward_history),
                    iteration,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length",
                    sum(episode_length_history) / len(episode_length_history),
                    iteration,
                )
            self._log_iteration(
                iteration, end_iteration, collection_time, learning_time,
                value_loss, surrogate_loss, reward_history,
                episode_length_history, ep_infos,
            )
            if iteration % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.model_dir, f"model_{iteration}.pt"))
                self.save(os.path.join(self.model_dir, "last.pt"))

        self.current_learning_iteration = end_iteration
        self.save(os.path.join(self.model_dir, f"model_{end_iteration}.pt"))
        self.save(os.path.join(self.model_dir, "last.pt"))

    def _log_iteration(
        self, iteration, end_iteration, collection_time, learning_time,
        value_loss, surrogate_loss, reward_history, episode_length_history,
        ep_infos, width=80, pad=35,
    ):
        iteration_time = collection_time + learning_time
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
        mean_std = self.actor_critic.std.mean().item()
        ep_string = ""
        if ep_infos:
            for key in ep_infos[0]:
                values = []
                for ep_info in ep_infos:
                    value = ep_info[key]
                    value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    values.append(value.reshape(-1).float().to(self.device))
                mean_value = torch.cat(values).mean().item()
                self.writer.add_scalar("Episode/" + key, mean_value, iteration)
                ep_string += f"{('Mean episode ' + key + ':'):>{pad}} {mean_value:.4f}\n"

        title = f" \033[1m Learning iteration {iteration + 1}/{end_iteration} \033[0m "
        log_string = (
            f"{'#' * width}\n"
            f"{title.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps} steps/s "
            f"(collection: {collection_time:.3f}s, learning {learning_time:.3f}s)\n"
            f"{'Value function loss:':>{pad}} {value_loss:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {surrogate_loss:.4f}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"
        )
        if reward_history:
            log_string += (
                f"{'Mean reward:':>{pad}} {sum(reward_history) / len(reward_history):.2f}\n"
                f"{'Mean episode length:':>{pad}} "
                f"{sum(episode_length_history) / len(episode_length_history):.2f}\n"
            )
        remaining = end_iteration - iteration - 1
        completed = self.current_learning_iteration
        eta = self.total_time / max(completed, 1) * remaining
        log_string += (
            ep_string
            + f"{'-' * width}\n"
            + f"{'Total timesteps:':>{pad}} {self.total_steps}\n"
            + f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            + f"{'Total time:':>{pad}} {self.total_time:.2f}s\n"
            + f"{'ETA:':>{pad}} {eta:.1f}s\n"
        )
        print(log_string, flush=True)

    def save(self, path):
        checkpoint = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iteration": self.current_learning_iteration,
            "total_steps": self.total_steps,
            "total_time": self.total_time,
        }
        temporary_path = path + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        checkpoint_iteration = checkpoint.get("iteration", 0)
        if checkpoint_iteration == 0:
            match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
            if match:
                checkpoint_iteration = int(match.group(1))
        self.current_learning_iteration = checkpoint_iteration
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_time = checkpoint.get("total_time", 0.0)

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)

        def policy(obs_dict):
            return self.actor_critic.act_inference(obs_dict["obs"])

        return policy
