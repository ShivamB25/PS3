"""Neural network architectures for the SAC agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from ..config.config import ModelConfig


def initialize_weights(module: nn.Module, gain: float = 1.0):
    """Initialize network weights using Xavier initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    """Actor network for the SAC agent with enhanced stability measures."""

    def __init__(self, state_dim: int, action_dim: int, config: ModelConfig):
        """Initialize the actor network."""
        super().__init__()

        # Smaller initial gain for better stability
        self.initial_gain = 0.1

        # Split temporal and non-temporal features
        self.temporal_features = 2  # DayOfWeek and Month
        self.non_temporal_dim = state_dim - self.temporal_features

        # Separate normalization for non-temporal features
        self.input_norm = nn.LayerNorm(self.non_temporal_dim)

        # Main network with temporal attention
        self.temporal_net = nn.Sequential(
            nn.Linear(self.temporal_features, config.actor_hidden_dims[0] // 4),
            nn.ReLU(),
        )

        self.feature_net = nn.Sequential(
            nn.Linear(self.non_temporal_dim, config.actor_hidden_dims[0] * 3 // 4),
            nn.LayerNorm(config.actor_hidden_dims[0] * 3 // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.combined_net = nn.Sequential(
            nn.Linear(config.actor_hidden_dims[0], config.actor_hidden_dims[1]),
            nn.LayerNorm(config.actor_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Output layers with smaller initialization
        self.mean = nn.Sequential(
            nn.Linear(config.actor_hidden_dims[-1], action_dim),
            nn.LayerNorm(action_dim),
        )

        self.log_std = nn.Sequential(
            nn.Linear(config.actor_hidden_dims[-1], action_dim),
            nn.LayerNorm(action_dim),
        )

        # Initialize weights with smaller values
        self.apply(lambda m: initialize_weights(m, gain=self.initial_gain))
        self.mean[0].apply(lambda m: initialize_weights(m, gain=0.005))
        self.log_std[0].apply(lambda m: initialize_weights(m, gain=0.005))

        # Initialize log_std bias for smaller initial variance
        self.log_std[0].bias.data.fill_(-3.0)

        # State validation bounds
        self.max_state_value = 5.0
        self.min_state_value = -5.0

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network with enhanced stability measures."""
        # Input validation
        if torch.isnan(state).any():
            raise ValueError(f"NaN detected in state input: {state}")

        # Split temporal and non-temporal features
        temporal_features = state[:, -self.temporal_features :]
        non_temporal_features = state[:, : -self.temporal_features]

        # Process temporal features
        temporal_out = self.temporal_net(temporal_features)

        # Process non-temporal features with normalization
        non_temporal_features = torch.clamp(
            non_temporal_features, self.min_state_value, self.max_state_value
        )
        features_out = self.feature_net(self.input_norm(non_temporal_features))

        # Combine features
        combined = torch.cat([features_out, temporal_out], dim=1)
        net_out = self.combined_net(combined)

        # Additional stability check
        if torch.isnan(net_out).any():
            raise ValueError(f"NaN detected in network output: {net_out}")

        # Generate mean and log_std with tight bounds
        mean = self.mean(net_out)
        log_std = self.log_std(net_out)

        # Constrain mean to prevent extreme values
        mean = torch.tanh(mean)  # Bound to [-1, 1]

        # Constrain log_std more tightly for better stability
        log_std = torch.clamp(log_std, min=-10.0, max=0.5)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the policy with enhanced numerical stability."""
        try:
            mean, log_std = self.forward(state)

            # Compute standard deviation with minimum bound
            std = torch.exp(log_std).clamp(min=1e-3)

            # Create normal distribution with error checking
            if torch.isnan(mean).any() or torch.isnan(std).any():
                raise ValueError("NaN detected before creating distribution")

            normal = torch.distributions.Normal(mean, std)

            # Sample with gradient using reparameterization
            x_t = normal.rsample()

            # Clip sampled values for stability
            x_t = torch.clamp(x_t, -10.0, 10.0)

            # Apply bounded squashing function
            action = torch.tanh(x_t)

            # Compute log probability with numerical stability
            log_prob = normal.log_prob(x_t)

            # Squashing correction with safe epsilon
            epsilon = 1e-6
            log_prob -= torch.log(1 - action.pow(2) + epsilon)

            # Sum log probabilities and ensure finite values
            log_prob = log_prob.sum(1, keepdim=True)

            # Final validation
            if torch.isnan(action).any() or torch.isnan(log_prob).any():
                raise ValueError("NaN detected in final output")

            return action, log_prob

        except Exception as e:
            print(f"Error in sample method: {str(e)}")
            print(
                f"State stats - min: {state.min()}, max: {state.max()}, mean: {state.mean()}"
            )
            raise e


class Critic(nn.Module):
    """Critic network for the SAC agent with enhanced stability measures."""

    def __init__(self, state_dim: int, action_dim: int, config: ModelConfig):
        """Initialize the critic network."""
        super().__init__()

        # Input dimensions and normalization
        input_dim = state_dim + action_dim
        self.input_norm = nn.LayerNorm(input_dim)

        # Smaller initial gain for stability
        self.initial_gain = 0.1

        # Q1 architecture with residual connections
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim, config.critic_hidden_dims[0]),
            nn.LayerNorm(config.critic_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.critic_hidden_dims[0], config.critic_hidden_dims[1]),
            nn.LayerNorm(config.critic_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.q1_out = nn.Sequential(
            nn.Linear(config.critic_hidden_dims[1], 1), nn.Tanh()  # Bound Q-values
        )

        # Q2 architecture with residual connections
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim, config.critic_hidden_dims[0]),
            nn.LayerNorm(config.critic_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.critic_hidden_dims[0], config.critic_hidden_dims[1]),
            nn.LayerNorm(config.critic_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.q2_out = nn.Sequential(
            nn.Linear(config.critic_hidden_dims[1], 1), nn.Tanh()  # Bound Q-values
        )

        # Initialize weights with smaller values
        self.apply(lambda m: initialize_weights(m, gain=self.initial_gain))
        self.q1_out[0].apply(lambda m: initialize_weights(m, gain=0.005))
        self.q2_out[0].apply(lambda m: initialize_weights(m, gain=0.005))

        # Value bounds for stability
        self.max_value = 5.0
        self.min_value = -5.0

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q-networks with enhanced stability measures."""
        try:
            # Input validation
            if torch.isnan(state).any() or torch.isnan(action).any():
                raise ValueError("NaN detected in inputs")

            # Concatenate and normalize inputs
            sa = torch.cat([state, action], 1)
            sa = torch.clamp(sa, self.min_value, self.max_value)
            sa = self.input_norm(sa)

            # Q1 forward pass
            q1_features = self.q1_net(sa)
            q1 = self.q1_out(q1_features)

            # Q2 forward pass
            q2_features = self.q2_net(sa)
            q2 = self.q2_out(q2_features)

            # Scale Q-values to reasonable range
            q1 = q1 * self.max_value
            q2 = q2 * self.max_value

            # Validation checks
            if torch.isnan(q1).any() or torch.isnan(q2).any():
                raise ValueError(f"NaN detected in Q-values: Q1={q1}, Q2={q2}")

            if (torch.abs(q1) > 1e3).any() or (torch.abs(q2) > 1e3).any():
                print(
                    f"Warning: Large Q-values detected: max_q1={q1.abs().max()}, max_q2={q2.abs().max()}"
                )
                q1 = torch.clamp(q1, -1e3, 1e3)
                q2 = torch.clamp(q2, -1e3, 1e3)

            return q1, q2

        except Exception as e:
            print(f"Error in critic forward pass: {str(e)}")
            print(
                f"State stats - min: {state.min()}, max: {state.max()}, mean: {state.mean()}"
            )
            print(
                f"Action stats - min: {action.min()}, max: {action.max()}, mean: {action.mean()}"
            )
            raise e

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through just Q1 network with stability measures."""
        # Concatenate and normalize inputs
        sa = torch.cat([state, action], 1)
        sa = torch.clamp(sa, self.min_value, self.max_value)
        sa = self.input_norm(sa)

        # Q1 forward pass with scaling
        q1_features = self.q1_net(sa)
        q1 = self.q1_out(q1_features) * self.max_value

        return torch.clamp(q1, -1e3, 1e3)
