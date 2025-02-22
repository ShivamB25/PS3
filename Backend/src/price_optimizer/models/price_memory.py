"""Price memory module for tracking explored price points."""

import numpy as np
from typing import Tuple, List, Optional
import torch


class PriceMemory:
    """Tracks explored price points and guides exploration."""

    def __init__(
        self,
        price_history: np.ndarray,
        std_multiplier: float = 2.0,
        num_bins: Optional[int] = None,
        decay_factor: Optional[float] = None,
        exploration_threshold: Optional[float] = None,
    ):
        """Initialize price memory dynamically based on historical data.

        Args:
            price_history: Array of historical prices to determine ranges
            std_multiplier: Number of standard deviations for price range
            num_bins: Optional number of bins (default: auto-calculated)
            decay_factor: Optional memory decay rate (default: data-based)
            exploration_threshold: Optional threshold (default: data-based)
        """
        # Analyze price history to set dynamic ranges
        self.price_mean = np.mean(price_history)
        self.price_std = np.std(price_history)

        # Set price bounds based on historical data
        self.min_price = max(0, self.price_mean - std_multiplier * self.price_std)
        self.max_price = self.price_mean + std_multiplier * self.price_std

        # Auto-calculate optimal number of bins if not provided
        if num_bins is None:
            # Use Freedman-Diaconis rule for bin width
            iqr = np.percentile(price_history, 75) - np.percentile(price_history, 25)
            bin_width = 2 * iqr / (len(price_history) ** (1 / 3))
            self.num_bins = max(
                10, min(100, int((self.max_price - self.min_price) / bin_width))
            )
        else:
            self.num_bins = num_bins

        # Calculate decay factor based on data frequency if not provided
        if decay_factor is None:
            # Slower decay for sparse data, faster for dense data
            self.decay_factor = 1 - (1 / len(price_history))
        else:
            self.decay_factor = decay_factor

        # Set exploration threshold based on price variance if not provided
        if exploration_threshold is None:
            # More exploration for high variance, less for low variance
            normalized_std = self.price_std / self.price_mean
            self.exploration_threshold = min(0.5, max(0.1, normalized_std))
        else:
            self.exploration_threshold = exploration_threshold

        # Initialize tracking arrays
        self.bin_edges = np.linspace(self.min_price, self.max_price, self.num_bins + 1)
        self.visit_counts = np.zeros(self.num_bins)
        self.success_metrics = np.zeros(self.num_bins)

        # Initialize with historical data
        for price in price_history:
            if self.min_price <= price <= self.max_price:
                bin_idx = self._get_bin_index(price)
                self.visit_counts[bin_idx] += 0.1  # Soft initialization

        # Track unexplored regions
        self.unexplored_mask = self.visit_counts < 0.5

        # Store statistics for adaptive exploration
        self.total_updates = 0
        self.recent_rewards = []
        self.exploration_rate = 1.0  # Start with high exploration

    def update(self, price: float, reward: float):
        """Update memory with new price point and reward."""
        bin_idx = self._get_bin_index(price)

        # Update visit counts with adaptive decay
        self.visit_counts *= self.decay_factor
        self.visit_counts[bin_idx] += 1

        # Update success metrics with adaptive learning rate
        visits = self.visit_counts[bin_idx]
        alpha = 1 / (1 + visits)  # Decreasing learning rate with more visits
        self.success_metrics[bin_idx] = (1 - alpha) * self.success_metrics[
            bin_idx
        ] + alpha * reward

        # Update exploration mask
        self.unexplored_mask[bin_idx] = False

        # Update exploration statistics
        self.total_updates += 1
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:  # Rolling window
            self.recent_rewards.pop(0)

        # Adapt exploration rate based on recent performance
        if len(self.recent_rewards) >= 10:
            recent_mean = np.mean(self.recent_rewards[-10:])
            overall_mean = np.mean(self.recent_rewards)
            if recent_mean > overall_mean:
                # Reduce exploration if recent performance is good
                self.exploration_rate *= 0.995
            else:
                # Increase exploration if recent performance is poor
                self.exploration_rate = min(1.0, self.exploration_rate * 1.005)

    def get_exploration_bonus(self, price: float) -> float:
        """Calculate dynamic exploration bonus for a price point."""
        bin_idx = self._get_bin_index(price)

        # Calculate visit-based bonus
        max_visits = max(1, np.max(self.visit_counts))
        visit_bonus = np.exp(-self.visit_counts[bin_idx] / max_visits)

        # Calculate region-based bonus
        region_bonus = 1.0 if self.unexplored_mask[bin_idx] else 0.0

        # Calculate success-based bonus
        success_bonus = 0.0
        if self.visit_counts[bin_idx] > 0:
            normalized_success = (
                self.success_metrics[bin_idx] - np.min(self.success_metrics)
            ) / (np.max(self.success_metrics) - np.min(self.success_metrics) + 1e-6)
            success_bonus = (
                1 - normalized_success
            )  # Explore less successful regions more

        # Combine bonuses with adaptive weights
        total_bonus = (
            0.4 * visit_bonus + 0.4 * region_bonus + 0.2 * success_bonus
        ) * self.exploration_rate

        return float(total_bonus)

    def get_optimal_exploration_std(self, price: float) -> float:
        """Calculate optimal standard deviation for exploration."""
        bin_idx = self._get_bin_index(price)

        # Base std on price distribution and exploration needs
        base_std = self.price_std * 0.1  # Start with 10% of price std

        # Adjust based on exploration status
        if self.unexplored_mask[bin_idx]:
            # Larger std for unexplored regions
            return base_std * 2.0 * self.exploration_rate

        # Find nearest unexplored region
        nearest_unexplored = self._find_nearest_unexplored(bin_idx)
        if nearest_unexplored is not None:
            distance = abs(nearest_unexplored - bin_idx)
            distance_factor = np.exp(-distance / (self.num_bins * 0.1))
            return base_std * (1.0 + distance_factor * self.exploration_rate)

        # Smaller std for well-explored regions
        return base_std * self.exploration_rate

    def get_promising_regions(self) -> List[Tuple[float, float]]:
        """Identify promising price regions based on success metrics."""
        # Dynamically determine threshold based on success distribution
        threshold = np.mean(self.success_metrics) + 0.5 * np.std(self.success_metrics)
        promising_mask = self.success_metrics > threshold

        # Group consecutive promising bins
        regions = []
        start_idx = None

        for i in range(self.num_bins):
            if promising_mask[i] and start_idx is None:
                start_idx = i
            elif not promising_mask[i] and start_idx is not None:
                regions.append((self.bin_edges[start_idx], self.bin_edges[i]))
                start_idx = None

        # Handle last region
        if start_idx is not None:
            regions.append((self.bin_edges[start_idx], self.bin_edges[-1]))

        return regions

    def _get_bin_index(self, price: float) -> int:
        """Get bin index for a price point."""
        bin_idx = np.digitize(price, self.bin_edges) - 1
        return min(max(bin_idx, 0), self.num_bins - 1)

    def _find_nearest_unexplored(self, current_idx: int) -> Optional[int]:
        """Find nearest unexplored bin index."""
        if not np.any(self.unexplored_mask):
            return None

        distances = np.abs(np.arange(self.num_bins) - current_idx)
        unexplored_distances = distances[self.unexplored_mask]
        if len(unexplored_distances) == 0:
            return None

        return int(np.argmin(unexplored_distances))
