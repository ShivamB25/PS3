"""Standalone test for P2P price memory functionality."""
import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Minimal type definitions
class NetworkMode(Enum):
    PRIVATE = "private"
    CONSORTIUM = "consortium"
    PUBLIC = "public"

@dataclass
class NetworkStatus:
    connected: bool
    mode: NetworkMode
    peer_count: int
    last_sync: float
    health_status: str

@dataclass
class PrivacyConfig:
    anonymize_data: bool = True
    encrypt_connection: bool = True
    data_sharing: Dict[str, str] = None
    encryption_level: str = "high"

@dataclass
class NetworkConfig:
    mode: NetworkMode
    company_id: Optional[str] = None
    privacy: PrivacyConfig = None

@dataclass
class MarketInsight:
    price_range: Dict[str, float]
    trend: float
    confidence: float
    timestamp: float
    source_type: str

# Mock PriceMemory for testing
class MockPriceMemory:
    def __init__(self, price_history: np.ndarray):
        self.price_mean = np.mean(price_history)
        self.price_std = np.std(price_history)
        self.recent_rewards = []
        self.visit_counts = np.zeros(10)
        self.bin_edges = np.linspace(
            self.price_mean - 2*self.price_std,
            self.price_mean + 2*self.price_std,
            11
        )

    def update(self, price: float, reward: float):
        bin_idx = self._get_bin_index(price)
        self.visit_counts[bin_idx] += 1
        self.recent_rewards.append(reward)

    def get_exploration_bonus(self, price: float) -> float:
        return 0.5  # Mock value

    def get_optimal_exploration_std(self, price: float) -> float:
        return self.price_std * 0.1

    def get_promising_regions(self) -> List[Tuple[float, float]]:
        return [(self.price_mean - self.price_std, self.price_mean + self.price_std)]

    def _get_bin_index(self, price: float) -> int:
        bin_idx = np.digitize(price, self.bin_edges) - 1
        return min(max(bin_idx, 0), len(self.visit_counts) - 1)

# Mock Network for testing
class MockNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.status = NetworkStatus(
            connected=False,
            mode=config.mode,
            peer_count=0,
            last_sync=0,
            health_status="initializing"
        )
        self.message_handlers = {}
        self.logger = logging.getLogger("mock_network")

    async def connect(self):
        self.status.connected = True
        self.status.health_status = "healthy"
        self.logger.info("Connected to mock network")

    async def disconnect(self):
        self.status.connected = False
        self.status.health_status = "disconnected"
        self.logger.info("Disconnected from mock network")

    async def broadcast_market_insight(self, insight: MarketInsight):
        self.logger.info(f"Broadcasting insight: {insight}")

    async def sync(self):
        self.status.last_sync = time.time()
        self.logger.info("Network synced")

    def register_message_handler(self, message_type: str, handler):
        self.message_handlers[message_type] = handler

# P2P Price Memory implementation
class P2PPriceMemory:
    def __init__(
        self,
        price_history: np.ndarray,
        network: MockNetwork,
        local_weight: float = 0.7,
        network_weight: float = 0.3,
        sync_interval: float = 60.0
    ):
        self.local_memory = MockPriceMemory(price_history)
        self.network = network
        self.local_weight = local_weight
        self.network_weight = network_weight
        self.sync_interval = sync_interval
        self.last_sync = 0.0
        self.network_insights = {}
        self.logger = logging.getLogger("p2p_memory")

        network.register_message_handler(
            "market_insight",
            self._handle_market_insight
        )

    async def update(self, price: float, reward: float):
        self.local_memory.update(price, reward)
        insight = self._prepare_market_insight(price, reward)
        
        if self.network.status.connected:
            await self.network.broadcast_market_insight(insight)
        
        await self._maybe_sync()

    def get_exploration_bonus(self, price: float) -> float:
        local_bonus = self.local_memory.get_exploration_bonus(price)
        network_bonus = 0.5  # Mock network bonus
        return self.local_weight * local_bonus + self.network_weight * network_bonus

    def get_optimal_exploration_std(self, price: float) -> float:
        local_std = self.local_memory.get_optimal_exploration_std(price)
        network_std = self.local_memory.price_std * 0.1  # Mock network std
        return self.local_weight * local_std + self.network_weight * network_std

    def get_promising_regions(self) -> List[Tuple[float, float]]:
        return self.local_memory.get_promising_regions()

    async def _maybe_sync(self):
        now = time.time()
        if now - self.last_sync >= self.sync_interval:
            await self.network.sync()
            self.last_sync = now

    def _prepare_market_insight(self, price: float, reward: float) -> MarketInsight:
        return MarketInsight(
            price_range={
                "min": float(price - self.local_memory.price_std),
                "max": float(price + self.local_memory.price_std)
            },
            trend=float(np.mean(self.local_memory.recent_rewards[-10:]) if self.local_memory.recent_rewards else 0.0),
            confidence=0.8,  # Mock confidence
            timestamp=time.time(),
            source_type=self.network.config.mode.value
        )

    async def _handle_market_insight(self, message: Dict):
        try:
            insight = MarketInsight(**message["data"])
            self.network_insights[message["metadata"]["peer_id"]] = insight
        except Exception as e:
            self.logger.error(f"Failed to handle market insight: {e}")

async def main():
    """Test P2P price memory functionality."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("memory_test")

    try:
        # Create test data
        price_history = np.array([10.0, 12.0, 15.0, 11.0, 13.0])

        # Create network configuration
        config = NetworkConfig(
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

        # Initialize components
        network = MockNetwork(config)
        memory = P2PPriceMemory(
            price_history=price_history,
            network=network,
            sync_interval=5.0
        )

        # Test workflow
        logger.info("Starting test workflow")
        
        await network.connect()
        logger.info("Network connected")

        test_price = 14.0
        test_reward = 0.8
        
        await memory.update(test_price, test_reward)
        logger.info("Memory updated")

        bonus = memory.get_exploration_bonus(test_price)
        logger.info(f"Exploration bonus: {bonus}")

        std = memory.get_optimal_exploration_std(test_price)
        logger.info(f"Optimal std: {std}")

        regions = memory.get_promising_regions()
        logger.info(f"Promising regions: {regions}")

        await network.disconnect()
        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
