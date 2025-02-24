"""P2P networking module for price optimization."""

from .types import (
    NetworkMode,
    NetworkConfig,
    NetworkStatus,
    NetworkMessage,
    MarketInsight,
    PrivacyConfig
)
from .config import NetworkConfigManager, ConfigurationError
from .network import PearNetwork, NetworkError

__all__ = [
    # Network types
    "NetworkMode",
    "NetworkConfig",
    "NetworkStatus",
    "NetworkMessage",
    "MarketInsight",
    "PrivacyConfig",
    
    # Configuration
    "NetworkConfigManager",
    "ConfigurationError",
    
    # Network implementation
    "PearNetwork",
    "NetworkError"
]

__version__ = "0.1.0"
