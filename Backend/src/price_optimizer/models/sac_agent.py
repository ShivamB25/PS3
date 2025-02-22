"""Soft Actor-Critic (SAC) agent implementation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from collections import deque
import random

from .networks import Actor, Critic
from ..config.config import ModelConfig


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        """Initialize buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        transitions = random.sample(list(self.buffer), batch_size)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.FloatTensor(np.array(batch[1]))
        rewards = torch.FloatTensor(np.array(batch[2]).reshape(-1, 1))
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(np.array(batch[4]).reshape(-1, 1))

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class SACAgent:
    """Soft Actor-Critic agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: ModelConfig,
        price_history: np.ndarray,
    ):
        """Initialize the SAC agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Model configuration
            price_history: Historical price data for initializing price memory
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = 0.2  # Temperature parameter for entropy
        self.target_entropy = -action_dim  # Heuristic value

        # Import price memory
        from .price_memory import PriceMemory

        # Initialize price memory with historical data
        self.price_memory = PriceMemory(price_history=price_history)

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, config)
        self.critic = Critic(state_dim, action_dim, config)
        self.critic_target = Critic(state_dim, action_dim, config)

        # Initialize target network with same weights
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

        # Initialize automatic entropy tuning
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)

        # Training info
        self.total_steps = 0
        self.episode_reward = 0
        self.training_info = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "rewards": [],
            "exploration_bonuses": [],
        }

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select an action given the current state.

        Args:
            state: Current state (flattened)
            evaluate: Whether to evaluate (no noise) or explore

        Returns:
            Selected action
        """
        # Ensure state is flattened and convert to tensor
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                # Get base action from actor
                mean, log_std = self.actor(state)

                # Get optimal exploration std from price memory
                price = mean.cpu().numpy()[0][0]  # First action dimension is price
                exploration_std = self.price_memory.get_optimal_exploration_std(price)

                # Apply exploration bonus to std
                bonus = self.price_memory.get_exploration_bonus(price)

                # Convert exploration_std to tensor and match device/shape
                exploration_std_tensor = torch.tensor(
                    exploration_std, device=mean.device
                ).expand_as(log_std)

                # Combine base network std with exploration std
                base_std = torch.exp(log_std)
                combined_std = base_std * (1.0 + bonus) + exploration_std_tensor

                # Create normal distribution with combined std
                normal = torch.distributions.Normal(mean, combined_std)

                # Sample with reparameterization
                x_t = normal.rsample()
                action = torch.tanh(x_t)

                # Store exploration bonus for logging
                self.training_info["exploration_bonuses"].append(float(bonus))

        return action.cpu().numpy()[0]

    def train(self, batch_size: int) -> Dict[str, float]:
        """Update the networks using a batch of experiences."""
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        # Update price memory for each transition
        for i in range(batch_size):
            price = float(actions[i][0])  # First action dimension is price
            reward = float(rewards[i])
            self.price_memory.update(price, reward)

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            # Add exploration bonus to rewards based on price memory
            exploration_bonuses = torch.tensor(
                [
                    self.price_memory.get_exploration_bonus(float(a[0]))
                    for a in next_actions
                ],
                device=rewards.device,
            ).unsqueeze(1)

            # Modify rewards with exploration bonus
            augmented_rewards = (
                rewards + 0.1 * exploration_bonuses
            )  # Small bonus weight

            value_target = augmented_rewards + (1 - dones) * self.gamma * (
                q_next - self.alpha * next_log_probs
            )

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, value_target) + F.mse_loss(q2, value_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update actor with exploration-aware objective
        actions_new, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)

        # Add exploration bonus to actor objective
        exploration_values = torch.tensor(
            [self.price_memory.get_exploration_bonus(float(a[0])) for a in actions_new],
            device=q_new.device,
        ).unsqueeze(1)

        # Modified actor loss incorporating exploration
        actor_loss = (self.alpha * log_probs - q_new - 0.1 * exploration_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update temperature parameter alpha with exploration awareness
        target_entropy = self.target_entropy * (
            1.0 + 0.1 * exploration_values.mean().item()
        )
        alpha_loss = -(self.log_alpha * (log_probs + target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # Update target networks with soft update
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Store training info
        self.training_info["actor_loss"].append(actor_loss.item())
        self.training_info["critic_loss"].append(critic_loss.item())
        self.training_info["alpha_loss"].append(alpha_loss.item())

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "mean_exploration_bonus": float(exploration_values.mean()),
        }

    def save(self, path: str):
        """Save the agent's networks."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "training_info": self.training_info,
            },
            path,
        )

    def load(self, path: str):
        """Load the agent's networks."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        self.log_alpha = checkpoint["log_alpha"]
        self.training_info = checkpoint["training_info"]
