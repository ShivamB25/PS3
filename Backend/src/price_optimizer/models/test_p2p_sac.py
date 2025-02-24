"""Test P2P-enhanced SAC agent functionality."""
import asyncio
import logging
import numpy as np
import torch
from pathlib import Path

from ..config.config import ModelConfig
from ..p2p.network import PearNetwork
from ..p2p.types import NetworkMode, NetworkConfig, PrivacyConfig
from .p2p_sac_agent import P2PSACAgent

class MockModelConfig:
    """Mock model configuration for testing."""
    def __init__(self):
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_learning_rate = 3e-4
        self.critic_learning_rate = 3e-4
        self.buffer_size = 1000000

async def main():
    """Test P2P SAC agent functionality."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("p2p_sac_test")

    try:
        # Create test environment
        state_dim = 3  # price, demand, time
        action_dim = 1  # price adjustment
        price_history = np.array([10.0, 12.0, 15.0, 11.0, 13.0])

        # Create network configuration
        network_config = NetworkConfig(
            mode=NetworkMode.PRIVATE,
            company_id="test_company",
            privacy=PrivacyConfig(
                data_sharing={
                    "price_data": "ranges_only",
                    "sales_data": "aggregated",
                    "trends": "full"
                }
            )
        )

        # Initialize network
        network = PearNetwork(network_config)
        logger.info("Network initialized")

        # Connect to network
        await network.connect()
        logger.info("Connected to network")

        # Initialize P2P SAC agent
        agent = P2PSACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=MockModelConfig(),
            price_history=price_history,
            network=network,
            local_weight=0.7,
            network_weight=0.3,
            sync_interval=5.0  # Short interval for testing
        )
        logger.info("P2P SAC agent initialized")

        # Test action selection
        test_state = np.array([12.0, 100.0, 0.5])  # price, demand, time
        action = await agent.select_action(test_state)
        logger.info(f"Selected action: {action}")

        # Test training
        # Create mock batch
        states = torch.randn(32, state_dim)
        actions = torch.randn(32, action_dim)
        rewards = torch.randn(32, 1)
        next_states = torch.randn(32, state_dim)
        dones = torch.zeros(32, 1)

        # Add to replay buffer
        for i in range(32):
            agent.replay_buffer.push(
                states[i].numpy(),
                actions[i].numpy(),
                rewards[i].item(),
                next_states[i].numpy(),
                dones[i].item()
            )

        # Train for a few steps
        for i in range(5):
            train_info = await agent.train(batch_size=16)
            logger.info(f"Training step {i+1}:")
            logger.info(f"  Actor loss: {train_info['actor_loss']:.4f}")
            logger.info(f"  Critic loss: {train_info['critic_loss']:.4f}")
            logger.info(f"  Alpha loss: {train_info['alpha_loss']:.4f}")

        # Test model insight sharing
        test_state = torch.randn(state_dim)
        test_action = torch.randn(action_dim)
        await agent._share_model_insight(test_state, test_action)
        logger.info("Model insight shared")

        # Test network sync
        await agent._maybe_sync()
        logger.info("Network synced")

        # Test save/load
        save_path = "test_p2p_sac.pt"
        agent.save(save_path)
        logger.info("Agent state saved")

        new_agent = P2PSACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=MockModelConfig(),
            price_history=price_history,
            network=network
        )
        new_agent.load(save_path)
        logger.info("Agent state loaded")

        # Cleanup
        Path(save_path).unlink()
        await network.disconnect()
        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
