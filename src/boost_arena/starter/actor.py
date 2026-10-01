"""The policy network, with the rulebook's action mask applied inside the actor.

The actor receives the 53 observation numbers followed by the 90 mask flags. Only the 53
go through the network; the mask removes the actions the car can't take before an action
is drawn. The network on its own, 53 in and 90 out, is what gets exported as a submission.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from rlgym_learn_algos.ppo import ActorCritic, BasicCritic, SeparateActorCritic
from rlgym_learn_algos.ppo.actor_critic.actor import Actor
from torch import nn

from .. import interface
from ..export import check_and_serialize, layers_from_sequential
from ..policy import DISABLED_LOGIT, MIN_PROBABILITY  # The same masking as the scorer uses


def make_network(layer_sizes, layer_norm=True, dtype=torch.float32):
    """Linear, layer norm and leaky ReLU per hidden layer, then a linear output layer."""
    layers = []
    previous = interface.OBS_SIZE
    for size in layer_sizes:
        layers.append(nn.Linear(previous, size, dtype=dtype))
        if layer_norm:
            layers.append(nn.LayerNorm(size, dtype=dtype))
        layers.append(nn.LeakyReLU())
        previous = size
    layers.append(nn.Linear(previous, interface.NUM_ACTIONS, dtype=dtype))
    return nn.Sequential(*layers)


class MaskedDiscreteFF(Actor):
    def __init__(self, layer_sizes, layer_norm, dtype, device):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.network = make_network(layer_sizes, layer_norm, dtype).to(device)

    def _split(self, obs_list):
        obs = obs_list if isinstance(obs_list, torch.Tensor) else torch.as_tensor(np.asarray(obs_list), dtype=self.dtype)
        obs = obs.to(self.device, dtype=self.dtype)
        return obs[:, : interface.OBS_SIZE], obs[:, interface.OBS_SIZE :]

    def probabilities(self, obs_list):
        obs, mask = self._split(obs_list)
        logits = self.network(obs) + DISABLED_LOGIT * (1.0 - mask)
        probabilities = torch.softmax(logits, dim=-1)
        return torch.clamp(probabilities, min=MIN_PROBABILITY, max=1.0)

    def get_actions(self, agent_id_list: Sequence, obs_list, **kwargs: dict[str, Any]):
        with torch.no_grad():
            probabilities = self.probabilities(obs_list)
            if kwargs.get("deterministic"):
                action = probabilities.argmax(dim=-1, keepdim=True)
                return action.cpu().numpy(), torch.zeros(action.shape[0])
            # Drawn on the processor: drawing on some accelerators is not dependable
            action = torch.multinomial(probabilities.cpu(), 1, True)
            log_prob = torch.log(probabilities.cpu()).gather(-1, action)
            return action.numpy(), log_prob.squeeze(-1)

    def get_backprop_data(self, agent_id_list: Sequence, obs_list, action_list, **kwargs: dict[str, Any]):
        probabilities = self.probabilities(obs_list)
        actions = action_list if isinstance(action_list, torch.Tensor) else torch.as_tensor(np.asarray(action_list))
        actions = actions.to(self.device).reshape(-1, 1)
        log_probabilities = torch.log(probabilities)
        entropy = -(log_probabilities * probabilities).sum(dim=-1)
        return log_probabilities.gather(-1, actions), entropy.mean()

    def export(self) -> bytes:
        """The network as a Boost Arena submission file."""
        return check_and_serialize(layers_from_sequential(self.network.cpu()), "leaky_relu")


def make_actor_critic(layer_sizes, layer_norm, critic_layer_sizes):
    def factory(obs_space, action_space, dtype, device, agent_controller) -> ActorCritic:
        actor = MaskedDiscreteFF(layer_sizes, layer_norm, dtype, device)
        critic = BasicCritic(obs_space[1], critic_layer_sizes, dtype, device)
        return SeparateActorCritic(actor, critic)

    return factory
