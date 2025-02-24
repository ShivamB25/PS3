"""P2P-enhanced price memory implementation."""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

from price_optimizer.p2p.network import PearNetwork
from price_optimizer.p2p.types import MarketInsight, NetworkMode
from price_optimizer.models.price_memory import PriceMemory

class P2PPriceMemory:
    """P2P-enhanced price memory with distributed insights."""

    def __init__(
        self,
        price_history: np.ndarray,
        network: PearNetwork,
        local_weight: float = 0.7,
        network_weight: float = 0.3,
        sync_interval: float = 60.0,  # Sync every minute
        **kwargs
    ):
        """Initialize P2P price memory.
        
        Args:
            price_history: Array of historical prices
            network: P2P network instance
            local_weight: Weight for local insights (0-1)
            network_weight: Weight for network insights (0-1)
            sync_interval: How often to sync with network (seconds)
            **kwargs: Additional arguments passed to PriceMemory
        """
        # Initialize local price memory
        self.local_memory = PriceMemory(price_history, **kwargs)
        
        # P2P components
        self.network = network
        self.logger = logging.getLogger("p2p_price_memory")
        
        # Weights for combining local and network insights
        assert local_weight + network_weight == 1.0, "Weights must sum to 1"
        self.local_weight = local_weight
        self.network_weight = network_weight
        
        # Network state
        self.network_insights: Dict[str, MarketInsight] = {}
        self.last_sync = 0.0
        self.sync_interval = sync_interval
        
        # Register network message handlers
        self.network.register_message_handler(
            "market_insight",
            self._handle_market_insight
        )

    async def update(self, price: float, reward: float):
        """Update price memory and share insights with network."""
        # Update local memory
        self.local_memory.update(price, reward)
        
        # Prepare market insight
        insight = self._prepare_market_insight(price, reward)
        
        # Share with network if connected
        if self.network.status.connected:
            try:
                await self.network.broadcast_market_insight(insight)
            except Exception as e:
                self.logger.error(f"Failed to broadcast insight: {e}")

        # Sync with network periodically
        await self._maybe_sync()

    def get_exploration_bonus(self, price: float) -> float:
        """Get exploration bonus combining local and network insights."""
        # Get local bonus
        local_bonus = self.local_memory.get_exploration_bonus(price)
        
        # Get network bonus
        network_bonus = self._calculate_network_bonus(price)
        
        # Combine insights
        return (
            self.local_weight * local_bonus +
            self.network_weight * network_bonus
        )

    def get_optimal_exploration_std(self, price: float) -> float:
        """Get optimal exploration std combining local and network insights."""
        # Get local std
        local_std = self.local_memory.get_optimal_exploration_std(price)
        
        # Get network std
        network_std = self._calculate_network_std(price)
        
        # Combine insights
        return (
            self.local_weight * local_std +
            self.network_weight * network_std
        )

    def get_promising_regions(self) -> List[Tuple[float, float]]:
        """Get promising regions combining local and network insights."""
        # Get local regions
        local_regions = self.local_memory.get_promising_regions()
        
        # Get network regions
        network_regions = self._get_network_promising_regions()
        
        # Combine and merge overlapping regions
        all_regions = local_regions + network_regions
        if not all_regions:
            return []
            
        # Sort regions by start price
        sorted_regions = sorted(all_regions, key=lambda x: x[0])
        
        # Merge overlapping regions
        merged = []
        current = sorted_regions[0]
        
        for next_region in sorted_regions[1:]:
            if current[1] >= next_region[0]:
                # Regions overlap, merge them
                current = (current[0], max(current[1], next_region[1]))
            else:
                # No overlap, add current region and start new one
                merged.append(current)
                current = next_region
                
        merged.append(current)
        return merged

    async def _maybe_sync(self):
        """Sync with network if enough time has passed."""
        now = time.time()
        if now - self.last_sync >= self.sync_interval:
            try:
                await self.network.sync()
                self.last_sync = now
            except Exception as e:
                self.logger.error(f"Network sync failed: {e}")

    def _prepare_market_insight(self, price: float, reward: float) -> MarketInsight:
        """Prepare market insight for network sharing."""
        # Get local statistics
        bin_idx = self.local_memory._get_bin_index(price)
        price_range = {
            "min": float(self.local_memory.bin_edges[bin_idx]),
            "max": float(self.local_memory.bin_edges[bin_idx + 1])
        }
        
        # Calculate trend from recent performance
        if self.local_memory.recent_rewards:
            trend = float(np.mean(self.local_memory.recent_rewards[-10:]))
        else:
            trend = 0.0
            
        # Calculate confidence based on visit counts
        confidence = float(
            self.local_memory.visit_counts[bin_idx] /
            max(1, np.max(self.local_memory.visit_counts))
        )
        
        return MarketInsight(
            price_range=price_range,
            trend=trend,
            confidence=confidence,
            timestamp=time.time(),
            source_type=self.network.config.mode.value
        )

    async def _handle_market_insight(self, message: Dict):
        """Handle incoming market insight from network."""
        try:
            insight = MarketInsight(**message["data"])
            self.network_insights[message["metadata"]["peer_id"]] = insight
        except Exception as e:
            self.logger.error(f"Failed to handle market insight: {e}")

    def _calculate_network_bonus(self, price: float) -> float:
        """Calculate exploration bonus from network insights."""
        if not self.network_insights:
            return 0.0
            
        # Find relevant insights for this price point
        relevant_insights = [
            insight for insight in self.network_insights.values()
            if insight.price_range["min"] <= price <= insight.price_range["max"]
        ]
        
        if not relevant_insights:
            return 1.0  # High bonus for unexplored price points
            
        # Calculate weighted average of trends and confidence
        total_weight = 0.0
        weighted_bonus = 0.0
        
        for insight in relevant_insights:
            # More recent insights get higher weight
            age = time.time() - insight.timestamp
            recency_weight = np.exp(-age / (24 * 3600))  # Decay over 24 hours
            
            # Higher confidence insights get higher weight
            weight = recency_weight * insight.confidence
            total_weight += weight
            
            # Lower trend suggests more room for exploration
            exploration_need = 1.0 - (insight.trend + 1) / 2
            weighted_bonus += weight * exploration_need
            
        return weighted_bonus / total_weight if total_weight > 0 else 1.0

    def _calculate_network_std(self, price: float) -> float:
        """Calculate optimal standard deviation from network insights."""
        if not self.network_insights:
            return self.local_memory.price_std * 0.1
            
        relevant_insights = [
            insight for insight in self.network_insights.values()
            if insight.price_range["min"] <= price <= insight.price_range["max"]
        ]
        
        if not relevant_insights:
            return self.local_memory.price_std * 0.2  # Higher std for unexplored areas
            
        # Calculate weighted average range
        total_weight = 0.0
        weighted_range = 0.0
        
        for insight in relevant_insights:
            age = time.time() - insight.timestamp
            recency_weight = np.exp(-age / (24 * 3600))
            weight = recency_weight * insight.confidence
            total_weight += weight
            
            price_range = insight.price_range["max"] - insight.price_range["min"]
            weighted_range += weight * price_range
            
        avg_range = weighted_range / total_weight if total_weight > 0 else self.local_memory.price_std
        return avg_range * 0.1  # Use 10% of range as std

    def _get_network_promising_regions(self) -> List[Tuple[float, float]]:
        """Identify promising regions from network insights."""
        if not self.network_insights:
            return []
            
        # Filter to recent insights with high confidence
        recent_insights = []
        for insight in self.network_insights.values():
            age = time.time() - insight.timestamp
            if age < 24 * 3600 and insight.confidence > 0.5:  # Last 24 hours
                recent_insights.append(insight)
                
        if not recent_insights:
            return []
            
        # Sort insights by trend
        sorted_insights = sorted(
            recent_insights,
            key=lambda x: x.trend,
            reverse=True
        )
        
        # Take top 30% as promising regions
        num_promising = max(1, int(len(sorted_insights) * 0.3))
        return [
            (insight.price_range["min"], insight.price_range["max"])
            for insight in sorted_insights[:num_promising]
        ]
