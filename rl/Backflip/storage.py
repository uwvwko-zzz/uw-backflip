import torch


class AsymmetricRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, horizon, actor_shape, critic_shape, action_shape, device):
        self.device = device
        self.horizon = horizon
        self.num_envs = num_envs
        self.step = 0
        self.observations = torch.zeros(horizon, num_envs, *actor_shape, device=device)
        self.critic_observations = torch.zeros(horizon, num_envs, *critic_shape, device=device)
        self.actions = torch.zeros(horizon, num_envs, *action_shape, device=device)
        self.rewards = torch.zeros(horizon, num_envs, 1, device=device)
        self.dones = torch.zeros(horizon, num_envs, 1, dtype=torch.uint8, device=device)
        self.values = torch.zeros(horizon, num_envs, 1, device=device)
        self.returns = torch.zeros_like(self.values)
        self.advantages = torch.zeros_like(self.values)
        self.actions_log_prob = torch.zeros_like(self.values)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)

    def add_transitions(self, tr):
        if self.step >= self.horizon:
            raise AssertionError("Rollout buffer overflow")
        dst = self.step
        self.observations[dst].copy_(tr.observations)
        self.critic_observations[dst].copy_(tr.critic_observations)
        self.actions[dst].copy_(tr.actions)
        self.rewards[dst].copy_(tr.rewards.view(-1, 1))
        self.dones[dst].copy_(tr.dones.view(-1, 1))
        self.values[dst].copy_(tr.values)
        self.actions_log_prob[dst].copy_(tr.actions_log_prob.view(-1, 1))
        self.mu[dst].copy_(tr.action_mean)
        self.sigma[dst].copy_(tr.action_sigma)
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0.0
        for step in reversed(range(self.horizon)):
            next_values = last_values if step == self.horizon - 1 else self.values[step + 1]
            next_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        batch_size = self.num_envs * self.horizon
        mini_batch_size = batch_size // num_mini_batches
        tensors = (
            self.observations.flatten(0, 1),
            self.critic_observations.flatten(0, 1),
            self.actions.flatten(0, 1),
            self.values.flatten(0, 1),
            self.advantages.flatten(0, 1),
            self.returns.flatten(0, 1),
            self.actions_log_prob.flatten(0, 1),
            self.mu.flatten(0, 1),
            self.sigma.flatten(0, 1),
        )
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for batch in range(num_mini_batches):
                ids = indices[batch * mini_batch_size:(batch + 1) * mini_batch_size]
                yield tuple(tensor[ids] for tensor in tensors)

    def clear(self):
        self.step = 0
