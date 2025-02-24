"""Network API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from ..p2p.network import PearNetwork
from ..p2p.types import NetworkMode, NetworkConfig, PrivacyConfig

router = APIRouter()

class NetworkStatusResponse(BaseModel):
    connected: bool
    mode: NetworkMode
    peer_count: int
    last_sync: float
    health_status: str

class DataSharingConfig(BaseModel):
    price_data: str
    sales_data: str
    trends: str

class NetworkConfigRequest(BaseModel):
    mode: NetworkMode
    privacy: Dict[str, bool | DataSharingConfig]

class MarketActivity(BaseModel):
    timestamp: float
    type: str
    value: float

class NetworkInsightResponse(BaseModel):
    insights: List[Dict]
    peer_count: int
    market_health: float
    recent_activity: List[MarketActivity]

# Global network instance
network: Optional[PearNetwork] = None

def get_network() -> PearNetwork:
    """Get or create network instance."""
    global network
    if network is None:
        config = NetworkConfig(
            mode=NetworkMode.PRIVATE,
            privacy=PrivacyConfig(
                anonymize_data=True,
                encrypt_connection=True,
                data_sharing={
                    "price_data": "ranges",
                    "sales_data": "aggregated",
                    "trends": "full"
                }
            )
        )
        network = PearNetwork(config)
    return network

@router.get("/status", response_model=NetworkStatusResponse)
async def get_network_status():
    """Get current network status."""
    try:
        net = get_network()
        return NetworkStatusResponse(
            connected=net.status.connected,
            mode=net.config.mode,
            peer_count=net.status.peer_count,
            last_sync=net.status.last_sync,
            health_status=net.status.health_status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_network_config():
    """Get current network configuration."""
    try:
        net = get_network()
        return {
            "mode": net.config.mode,
            "privacy": {
                "anonymize_data": net.config.privacy.anonymize_data,
                "encrypt_connection": net.config.privacy.encrypt_connection,
                "data_sharing": net.config.privacy.data_sharing
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
async def update_network_config(config: NetworkConfigRequest):
    """Update network configuration."""
    try:
        net = get_network()
        
        # Update network mode
        net.config.mode = config.mode
        
        # Update privacy settings
        if isinstance(config.privacy, dict):
            for key, value in config.privacy.items():
                if key == "data_sharing":
                    net.config.privacy.data_sharing.update(value)
                else:
                    setattr(net.config.privacy, key, value)
        
        # Reconnect with new config
        await net.disconnect()
        await net.connect()
        
        return {
            "status": "success",
            "message": "Network configuration updated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights", response_model=NetworkInsightResponse)
async def get_network_insights():
    """Get network-wide market insights."""
    try:
        net = get_network()
        insights = await net.get_market_insights()
        
        return NetworkInsightResponse(
            insights=insights.insights,
            peer_count=net.status.peer_count,
            market_health=insights.market_health,
            recent_activity=insights.recent_activity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/{product_id}")
async def get_product_insights(product_id: str):
    """Get market insights for specific product."""
    try:
        net = get_network()
        insights = await net.get_product_insights(product_id)
        
        return NetworkInsightResponse(
            insights=insights.insights,
            peer_count=net.status.peer_count,
            market_health=insights.market_health,
            recent_activity=insights.recent_activity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
