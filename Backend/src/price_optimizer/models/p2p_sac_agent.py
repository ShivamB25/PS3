"""P2P-enhanced Soft Actor-Critic (SAC) agent implementation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque
import random
import time
import json

from .networks import Actor, Critic
from .sac_agent import ReplayBuffer, SACAgent
from .p2p_price_memory import P2PPriceMemory
from ..config.config import ModelConfig
from ..p2p.network import PearNetwork
from ..p2p.types import NetworkMode, MarketInsight

class ModelInsight:
    """Structure for sharing model insights."""
    def __init__(
        self,
        state_value: float,
        action_value: float,
        entropy: float,
        timestamp: float,
        confidence: float
    ):
        self.state_value = state_value
        self.action_value = action_value
        self.entropy = entropy
        self.timestamp = timestamp
        self.confidence = confidence

class P2PSACAgent(SACAgent):
    """P2P-enhanced Soft Actor-Critic agent with collaborative learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: ModelConfig,
        price_history: np.ndarray,
        network: PearNetwork,
        local_weight: float = 0.7,
        network_weight: float = 0.3,
        sync_interval: float = 300.0,  # 5 minutes
    ):
        """Initialize P2P SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Model configuration
            price_history: Historical price data
            network: P2P network instance
            local_weight: Weight for local decisions
            network_weight: Weight for network insights
            sync_interval: How often to sync with network (seconds)
        """
        super().__init__(state_dim, action_dim, config, price_history)
        
        # Replace standard price memory with P2P version
        self.price_memory = P2PPriceMemory(
            price_history=price_history,
            network=network,
            local_weight=local_weight,
            network_weight=network_weight
        )
        
        # P2P components
        self.network = network
        self.local_weight = local_weight
        self.network_weight = network_weight
        self.last_sync = 0.0
        self.sync_interval = sync_interval
        self.network_insights: Dict[str, ModelInsight] = {}
        
        # Register network message handlers
        self.network.register_message_handler(
            "model_insight",
            self._handle_model_insight
        )
        
        # Enhanced training info
        self.training_info.update({
            "network_value_diff": [],
            "network_entropy_diff": [],
            "sync_counts": [],
        })

    async def select_action(
        self, 
        state: np.ndarray, 
        evaluate: bool = False
    ) -> np.ndarray:
        """Select action with network-enhanced exploration."""
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                # Get base action distribution
                mean, log_std = self.actor(state)
                
                # Get network-enhanced exploration parameters
                price = mean.cpu().numpy()[0][0]
                exploration_std = self.price_memory.get_optimal_exploration_std(price)
                bonus = self.price_memory.get_exploration_bonus(price)
                
                # Incorporate network insights into exploration
                network_std = self._calculate_network_std(state, price)
                network_bonus = self._calculate_network_bonus(state, price)
                
                # Combine local and network guidance
                final_std = (
                    self.local_weight * exploration_std +
                    self.network_weight * network_std
                )
                final_bonus = (
                    self.local_weight * bonus +
                    self.network_weight * network_bonus
                )
                
                # Create enhanced exploration distribution
                exploration_std_tensor = torch.tensor(
                    final_std,
                    device=mean.device
                ).expand_as(log_std)
                
                base_std = torch.exp(log_std)
                combined_std = base_std * (1.0 + final_bonus) + exploration_std_tensor
                
                # Sample action
                normal = torch.distributions.Normal(mean, combined_std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)
                
                # Store exploration info
                self.training_info["exploration_bonuses"].append(float(final_bonus))

        return action.cpu().numpy()[0]

    async def train(self, batch_size: int) -> Dict[str, float]:
        """Train with collaborative learning."""
        # Sample experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        # Update price memory
        for i in range(batch_size):
            price = float(actions[i][0])
            reward = float(rewards[i])
            await self.price_memory.update(price, reward)

        # Get network-enhanced value estimates
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # Enhance value estimates with network insights
            network_values = self._get_network_values(next_states, next_actions)
            enhanced_q_next = (
                self.local_weight * q_next +
                self.network_weight * network_values
            )
            
            # Calculate value targets
            value_target = rewards + (1 - dones) * self.gamma * (
                enhanced_q_next - self.alpha * next_log_probs
            )

        # Update critic
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, value_target) + F.mse_loss(q2, value_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 
            1.0
        )
        self.critic_optimizer.step()

        # Update actor with collaborative insights
        actions_new, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        # Enhance action values with network insights
        network_values = self._get_network_values(states, actions_new)
        enhanced_q_new = (
            self.local_weight * q_new +
            self.network_weight * network_values
        )
        
        # Calculate enhanced actor loss
        actor_loss = (
            self.alpha * log_probs - enhanced_q_new
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            1.0
        )
        self.actor_optimizer.step()

        # Update temperature parameter
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()

        # Soft update target network
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Share model insights if connected
        if self.network.status.connected:
            await self._share_model_insight(states[-1], actions_new[-1])

        # Periodic network sync
        await self._maybe_sync()

        # Store training info
        self.training_info["actor_loss"].append(actor_loss.item())
        self.training_info["critic_loss"].append(critic_loss.item())
        self.training_info["alpha_loss"].append(alpha_loss.item())
        
        if len(network_values) > 0:
            value_diff = (enhanced_q_new - q_new).mean().item()
            self.training_info["network_value_diff"].append(value_diff)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
        }

    def _calculate_network_std(
        self,
        state: torch.Tensor,
        price: float
    ) -> float:
        """Calculate exploration std from network insights."""
        if not self.network_insights:
            return self.price_memory.get_optimal_exploration_std(price)
            
        # Calculate weighted average of network insights
        total_weight = 0.0
        weighted_std = 0.0
        
        for insight in self.network_insights.values():
            # Weight by recency and confidence
            age = time.time() - insight.timestamp
            recency_weight = np.exp(-age / (24 * 3600))
            weight = recency_weight * insight.confidence
            
            # Higher entropy suggests more exploration needed
            exploration_need = insight.entropy
            weighted_std += weight * exploration_need
            total_weight += weight
            
        if total_weight > 0:
            return weighted_std / total_weight
        return self.price_memory.get_optimal_exploration_std(price)

    def _calculate_network_bonus(
        self,
        state: torch.Tensor,
        price: float
    ) -> float:
        """Calculate exploration bonus from network insights."""
        if not self.network_insights:
            return self.price_memory.get_exploration_bonus(price)
            
        total_weight = 0.0
        weighted_bonus = 0.0
        
        for insight in self.network_insights.values():
            age = time.time() - insight.timestamp
            recency_weight = np.exp(-age / (24 * 3600))
            weight = recency_weight * insight.confidence
            
            # Lower value suggests more exploration needed
            exploration_need = 1.0 - (insight.state_value + 1) / 2
            weighted_bonus += weight * exploration_need
            total_weight += weight
            
        if total_weight > 0:
            return weighted_bonus / total_weight
        return self.price_memory.get_exploration_bonus(price)

    def _get_network_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Get value estimates from network insights."""
        if not self.network_insights:
            return torch.zeros_like(states[:, 0]).unsqueeze(1)
            
        # Calculate weighted average of value estimates
        values = []
        for state, action in zip(states, actions):
            total_weight = 0.0
            weighted_value = 0.0
            
            for insight in self.network_insights.values():
                age = time.time() - insight.timestamp
                recency_weight = np.exp(-age / (24 * 3600))
                weight = recency_weight * insight.confidence
                
                weighted_value += weight * insight.action_value
                total_weight += weight
                
            if total_weight > 0:
                values.append(weighted_value / total_weight)
            else:
                values.append(0.0)
                
        return torch.tensor(values, device=states.device).unsqueeze(1)

    async def _share_model_insight(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ):
        """Share model insights with network."""
        with torch.no_grad():
            # Calculate state value
            q1, q2 = self.critic(
                state.unsqueeze(0),
                action.unsqueeze(0)
            )
            state_value = torch.min(q1, q2).item()
            
            # Calculate action value
            _, log_prob = self.actor.sample(state.unsqueeze(0))
            action_value = (
                state_value - self.alpha * log_prob.item()
            )
            
            # Calculate entropy
            entropy = -log_prob.item()
            
            # Create insight
            insight = ModelInsight(
                state_value=state_value,
                action_value=action_value,
                entropy=entropy,
                timestamp=time.time(),
                confidence=0.8  # TODO: Calculate based on model performance
            )
            
            # Share with network
            await self.network.broadcast_market_insight({
                "type": "model_insight",
                "data": insight.__dict__,
                "metadata": {
                    "network": self.network.config.mode.value,
                    "version": "1.0"
                }
            })

    async def _handle_model_insight(self, message: Dict):
        """Handle incoming model insight."""
        try:
            data = message["data"]
            insight = ModelInsight(
                state_value=data["state_value"],
                action_value=data["action_value"],
                entropy=data["entropy"],
                timestamp=data["timestamp"],
                confidence=data["confidence"]
            )
            self.network_insights[message["metadata"]["peer_id"]] = insight
        except Exception as e:
            print(f"Failed to handle model insight: {e}")

    async def _maybe_sync(self):
        """Periodic network synchronization."""
        now = time.time()
        if now - self.last_sync >= self.sync_interval:
            try:
                await self.network.sync()
                self.last_sync = now
                self.training_info["sync_counts"].append(1)
            except Exception as e:
                print(f"Network sync failed: {e}")
                self.training_info["sync_counts"].append(0)

    def save(self, path: str):
        """Save agent state with network insights."""
        state = super().save(path)
        # Add network-specific info
        state.update({
            "network_insights": {
                k: v.__dict__ for k, v in self.network_insights.items()
            },
            "last_sync": self.last_sync
        })
        torch.save(state, path)

    def load(self, path: str):
        """Load agent state with network insights."""
        super().load(path)
        checkpoint = torch.load(path)
        # Restore network-specific info
        if "network_insights" in checkpoint:
            self.network_insights = {
                k: ModelInsight(**v)
                for k, v in checkpoint["network_insights"].items()
            }
        if "last_sync" in checkpoint:
            self.last_sync = checkpoint["last_sync"]
