"""Type definitions for P2P components."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

class NetworkMode(Enum):
    """Network operation modes."""
    PRIVATE = "private"
    CONSORTIUM = "consortium" 
    PUBLIC = "public"

@dataclass
class PrivacyConfig:
    """Privacy settings configuration."""
    anonymize_data: bool = True
    encrypt_connection: bool = True
    data_sharing: Dict[str, str] = None
    encryption_level: str = "high"

    def __post_init__(self):
        if self.data_sharing is None:
            self.data_sharing = {
                "price_data": "ranges_only",
                "sales_data": "aggregated",
                "trends": "full"
            }

@dataclass
class NetworkConfig:
    """Network configuration settings."""
    mode: NetworkMode
    company_id: Optional[str] = None  # For private mode
    consortium_id: Optional[str] = None  # For consortium mode
    stores: Optional[List[str]] = None  # For private mode
    privacy: PrivacyConfig = None

    def __post_init__(self):
        if self.privacy is None:
            self.privacy = PrivacyConfig()

@dataclass
class NetworkMessage:
    """Network message structure."""
    type: str
    data: Dict
    timestamp: float
    signature: str
    metadata: Dict[str, str]

@dataclass
class NetworkStatus:
    """Network connection status."""
    connected: bool
    mode: NetworkMode
    peer_count: int
    last_sync: float
    health_status: str

@dataclass
class MarketInsight:
    """Market insight data structure."""
    price_range: Dict[str, float]
    trend: float
    confidence: float
    timestamp: float
    source_type: str  # private/consortium/public
