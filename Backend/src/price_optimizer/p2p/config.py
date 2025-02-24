"""Configuration management for P2P network."""
import os
from typing import Dict, Optional

from .types import NetworkMode, NetworkConfig, PrivacyConfig

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass

class NetworkConfigManager:
    """Manages P2P network configuration."""

    DEFAULT_PRIVATE_CONFIG = {
        "mode": NetworkMode.PRIVATE,
        "privacy": PrivacyConfig(
            anonymize_data=True,
            encrypt_connection=True,
            data_sharing={
                "price_data": "full",
                "sales_data": "full",
                "trends": "full"
            },
            encryption_level="high"
        )
    }

    DEFAULT_CONSORTIUM_CONFIG = {
        "mode": NetworkMode.CONSORTIUM,
        "privacy": PrivacyConfig(
            anonymize_data=True,
            encrypt_connection=True,
            data_sharing={
                "price_data": "ranges_only",
                "sales_data": "aggregated",
                "trends": "full"
            },
            encryption_level="high"
        )
    }

    DEFAULT_PUBLIC_CONFIG = {
        "mode": NetworkMode.PUBLIC,
        "privacy": PrivacyConfig(
            anonymize_data=True,
            encrypt_connection=True,
            data_sharing={
                "price_data": "ranges_only",
                "sales_data": "none",
                "trends": "aggregated"
            },
            encryption_level="maximum"
        )
    }

    @classmethod
    def create_config(cls, mode: NetworkMode, **kwargs) -> NetworkConfig:
        """Create a network configuration based on mode and optional parameters."""
        if mode == NetworkMode.PRIVATE:
            config_dict = cls.DEFAULT_PRIVATE_CONFIG.copy()
            if not kwargs.get("company_id"):
                raise ConfigurationError("company_id is required for private mode")
            config_dict.update(kwargs)
        
        elif mode == NetworkMode.CONSORTIUM:
            config_dict = cls.DEFAULT_CONSORTIUM_CONFIG.copy()
            if not kwargs.get("consortium_id"):
                raise ConfigurationError("consortium_id is required for consortium mode")
            config_dict.update(kwargs)
        
        elif mode == NetworkMode.PUBLIC:
            config_dict = cls.DEFAULT_PUBLIC_CONFIG.copy()
            config_dict.update(kwargs)
        
        else:
            raise ConfigurationError(f"Invalid network mode: {mode}")

        return NetworkConfig(**config_dict)

    @staticmethod
    def validate_config(config: NetworkConfig) -> bool:
        """Validate network configuration."""
        try:
            # Check mode-specific requirements
            if config.mode == NetworkMode.PRIVATE:
                if not config.company_id:
                    raise ConfigurationError("Private mode requires company_id")
            
            elif config.mode == NetworkMode.CONSORTIUM:
                if not config.consortium_id:
                    raise ConfigurationError("Consortium mode requires consortium_id")

            # Validate privacy settings
            if config.privacy:
                required_sharing_types = {"price_data", "sales_data", "trends"}
                if not all(k in config.privacy.data_sharing for k in required_sharing_types):
                    raise ConfigurationError("Missing required data sharing types")

                valid_sharing_levels = {"full", "ranges_only", "aggregated", "none"}
                if not all(v in valid_sharing_levels for v in config.privacy.data_sharing.values()):
                    raise ConfigurationError("Invalid data sharing level")

            return True

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")

    @staticmethod
    def load_from_env() -> NetworkConfig:
        """Load configuration from environment variables."""
        try:
            mode = NetworkMode(os.getenv("PEAR_NETWORK_MODE", "public").lower())
            config_kwargs = {
                "mode": mode,
                "company_id": os.getenv("PEAR_COMPANY_ID"),
                "consortium_id": os.getenv("PEAR_CONSORTIUM_ID"),
                "stores": os.getenv("PEAR_STORES", "").split(",") if os.getenv("PEAR_STORES") else None,
                "privacy": PrivacyConfig(
                    anonymize_data=os.getenv("PEAR_ANONYMIZE_DATA", "true").lower() == "true",
                    encrypt_connection=os.getenv("PEAR_ENCRYPT_CONNECTION", "true").lower() == "true",
                    encryption_level=os.getenv("PEAR_ENCRYPTION_LEVEL", "high")
                )
            }
            
            return NetworkConfigManager.create_config(**config_kwargs)
        
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from environment: {str(e)}")

    @staticmethod
    def save_config(config: NetworkConfig, path: str) -> None:
        """Save configuration to a file."""
        import json
        from dataclasses import asdict
        
        try:
            # Convert Enum to string for serialization
            config_dict = asdict(config)
            config_dict["mode"] = config.mode.value
            
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")

    @staticmethod
    def load_config(path: str) -> NetworkConfig:
        """Load configuration from a file."""
        import json
        
        try:
            with open(path, "r") as f:
                config_dict = json.load(f)
            
            # Convert string back to Enum
            config_dict["mode"] = NetworkMode(config_dict["mode"])
            
            return NetworkConfig(**config_dict)
        
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
