import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name):
    activations = {
        "elu": nn.ELU(), "relu": nn.ReLU(), "selu": nn.SELU(),
        "lrelu": nn.LeakyReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


def _mlp(input_dim, hidden_dims, output_dim, activation):
    layers = []
    dims = [input_dim, *hidden_dims]
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend((nn.Linear(in_dim, out_dim), _activation(activation)))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class AsymmetricActorCritic(nn.Module):
    """Actor uses deployable observations; critic uses simulator-only state."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        **_,
    ):
        super().__init__()
        self.actor = _mlp(num_actor_obs, actor_hidden_dims, num_actions, activation)
        self.critic = _mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, actor_obs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, actor_obs):
        return self.actor(actor_obs)

    def evaluate(self, critic_obs):
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass
