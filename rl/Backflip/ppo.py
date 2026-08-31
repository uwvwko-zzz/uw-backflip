import torch
import torch.nn as nn
import torch.optim as optim

from .storage import AsymmetricRolloutStorage


class AsymmetricPPO:
    def __init__(self, actor_critic, device="cpu", **cfg):
        self.actor_critic = actor_critic.to(device)
        self.device = device
        self.clip_param = cfg.get("clip_param", 0.2)
        self.gamma = cfg.get("gamma", 0.99)
        self.lam = cfg.get("lam", 0.95)
        self.value_loss_coef = cfg.get("value_loss_coef", 1.0)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.min_action_std = cfg.get("min_action_std", 0.0)
        self.max_action_std = cfg.get("max_action_std", float("inf"))
        self.learning_rate = cfg.get("learning_rate", 1e-3)
        self.max_grad_norm = cfg.get("max_grad_norm", 1.0)
        self.use_clipped_value_loss = cfg.get("use_clipped_value_loss", True)
        self.schedule = cfg.get("schedule", "adaptive")
        self.desired_kl = cfg.get("desired_kl", 0.01)
        self.num_learning_epochs = cfg.get("num_learning_epochs", 5)
        self.num_mini_batches = cfg.get("num_mini_batches", 4)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.transition = AsymmetricRolloutStorage.Transition()
        self.storage = None

    def init_storage(self, num_envs, horizon, actor_shape, critic_shape, action_shape):
        self.storage = AsymmetricRolloutStorage(
            num_envs, horizon, actor_shape, critic_shape, action_shape, self.device
        )

    def act(self, obs_dict):
        actor_obs = obs_dict["obs"]
        critic_obs = obs_dict["privileged_info"]
        self.transition.actions = self.actor_critic.act(actor_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = actor_obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, obs_dict):
        last_values = self.actor_critic.evaluate(obs_dict["privileged_info"]).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        value_total = surrogate_total = 0.0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for actor_obs, critic_obs, actions, old_values, advantages, returns, old_log_prob, old_mu, old_sigma in generator:
            self.actor_critic.update_distribution(actor_obs)
            log_prob = self.actor_critic.get_actions_log_prob(actions)
            values = self.actor_critic.evaluate(critic_obs)
            mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma / old_sigma + 1e-5)
                        + (old_sigma.square() + (old_mu - mu).square()) / (2.0 * sigma.square())
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > 2.0 * self.desired_kl:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < 0.5 * self.desired_kl:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(log_prob - old_log_prob.squeeze(-1))
            surrogate = -advantages.squeeze(-1) * ratio
            clipped = -advantages.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, clipped).mean()

            if self.use_clipped_value_loss:
                clipped_values = old_values + (values - old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (values - returns).square(), (clipped_values - returns).square()
                ).mean()
            else:
                value_loss = (returns - values).square().mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss \
                   - self.entropy_coef * self.actor_critic.entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                self.actor_critic.std.clamp_(
                    min=self.min_action_std, max=self.max_action_std
                )
            value_total += value_loss.item()
            surrogate_total += surrogate_loss.item()

        count = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return value_total / count, surrogate_total / count
