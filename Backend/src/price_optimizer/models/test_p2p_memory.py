"""Test P2P-enhanced price memory functionality."""

import asyncio
import logging
import numpy as np
from pathlib import Path

from price_optimizer.p2p.network import PearNetwork
from price_optimizer.p2p.config import NetworkConfigManager
from price_optimizer.p2p.types import NetworkMode, NetworkConfig, PrivacyConfig
from price_optimizer.models.p2p_price_memory import P2PPriceMemory

async def main():
    """Test P2P price memory functionality."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("p2p_memory_test")

    try:
        # Create test price history
        price_history = np.array([10.0, 12.0, 15.0, 11.0, 13.0])

        # Create network configuration
        config = NetworkConfig(
            mode=NetworkMode.PRIVATE,
            company_id="test_company",
            privacy=PrivacyConfig(
                anonymize_data=True,
                encrypt_connection=True,
                data_sharing={
                    "price_data": "ranges_only",
                    "sales_data": "aggregated",
                    "trends": "full"
                }
            )
        )

        # Initialize network
        network = PearNetwork(config)
        logger.info("Network initialized")

        # Connect to network
        await network.connect()
        logger.info("Connected to network")

        # Initialize P2P price memory
        memory = P2PPriceMemory(
            price_history=price_history,
            network=network,
            local_weight=0.7,
            network_weight=0.3,
            sync_interval=5.0  # Short interval for testing
        )
        logger.info("P2P price memory initialized")

        # Test update and insight sharing
        test_price = 14.0
        test_reward = 0.8
        await memory.update(test_price, test_reward)
        logger.info("Price memory updated and insight shared")

        # Test exploration bonus
        bonus = memory.get_exploration_bonus(test_price)
        logger.info(f"Exploration bonus for price {test_price}: {bonus}")

        # Test optimal exploration std
        std = memory.get_optimal_exploration_std(test_price)
        logger.info(f"Optimal exploration std for price {test_price}: {std}")

        # Test promising regions
        regions = memory.get_promising_regions()
        logger.info(f"Promising regions: {regions}")

        # Test network sync
        await memory._maybe_sync()
        logger.info("Network sync completed")

        # Disconnect from network
        await network.disconnect()
        logger.info("Disconnected from network")

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
