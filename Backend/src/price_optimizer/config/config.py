"""Configuration for the price optimization RL system."""

from dataclasses import dataclass, field
from typing import Dict, List
from functools import partial


@dataclass
class DataConfig:
    """Data-related configuration."""

    data_path: str = "data/woolballhistory.csv"
    history_window: int = 7  # Number of days to use as history
    train_test_split: float = 0.8
    features: List[str] = field(
        default_factory=lambda: [
            "Product Price",
            "Organic Conversion Percentage",
            "Ad Conversion Percentage",
            "Total Profit",
            "Total Sales",
            "Predicted Sales",
            "Price_MA7",
            "Sales_MA7",
            "DayOfWeek",
            "Month",
        ]
    )


@dataclass
class EnvConfig:
    """Environment configuration."""

    price_step: float = 0.05
    exploration_bonus_scale: float = 0.3
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "sales": 0.4,
            "price": 0.3,
            "organic_conversion": 0.15,
            "ad_conversion": 0.15,
        }
    )


@dataclass
class ModelConfig:
    """Model configuration."""

    # Actor Network
    actor_learning_rate: float = 3e-4
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Critic Network
    critic_learning_rate: float = 3e-4
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Training
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005  # Target network update rate


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_episodes: int = 1000
    steps_per_episode: int = 100
    eval_frequency: int = 10
    save_frequency: int = 50
    random_seed: int = 42
    wandb_project: str = "price-optimization"
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
