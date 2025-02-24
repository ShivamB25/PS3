"""Test script for P2P network functionality."""
import asyncio
import json
import logging
from pathlib import Path

from price_optimizer.p2p.types import NetworkMode, MarketInsight
from price_optimizer.p2p.config import NetworkConfigManager
from price_optimizer.p2p.network import PearNetwork

async def main():
    """Test basic P2P network functionality."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pear_test")

    try:
        # Load example configuration
        config_path = Path(__file__).parent / "example_config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        
        # Create network configuration
        config = NetworkConfigManager.create_config(
            mode=NetworkMode(config_dict["mode"]),
            **config_dict
        )
        
        # Initialize network
        network = PearNetwork(config)
        logger.info("Network initialized")
        
        # Connect to network
        await network.connect()
        logger.info("Connected to network")
        
        # Get network state
        state = await network.get_network_state()
        logger.info(f"Network state: {json.dumps(state, indent=2)}")
        
        # Create and broadcast a test market insight
        insight = MarketInsight(
            price_range={"min": 10.0, "max": 15.0},
            trend=0.5,
            confidence=0.8,
            timestamp=1234567890.0,
            source_type="test"
        )
        await network.broadcast_market_insight(insight)
        logger.info("Market insight broadcasted")
        
        # Sync with network
        await network.sync()
        logger.info("Network synchronized")
        
        # Disconnect
        await network.disconnect()
        logger.info("Disconnected from network")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
