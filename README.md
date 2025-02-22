# Price Optimization using Reinforcement Learning

A sophisticated price optimization system using Soft Actor-Critic (SAC) reinforcement learning to dynamically optimize product pricing while maximizing sales and conversion rates. The system learns from historical data to suggest optimal prices that balance revenue maximization with market dynamics.

## Features

### Core Capabilities
- Continuous price optimization using state-of-the-art SAC algorithm
- Intelligent exploration of price ranges with automatic entropy tuning
- Multi-objective reward system considering:
  - Sales performance (40% weight)
  - Price optimization (30% weight)
  - Organic conversion rates (15% weight)
  - Ad conversion rates (15% weight)
- Robust state management with 7-day historical windows
- Advanced neural network architectures with stability measures
- Comprehensive validation and error handling

### Technical Features
- Twin Q-networks for stable learning
- Automatic entropy adjustment
- Experience replay with prioritized sampling
- State and action validation
- Gradient clipping and normalization
- LayerNorm and Dropout for regularization

### Production Features
- Real-time price predictions
- Detailed performance reporting
- Comprehensive monitoring via Weights & Biases
- Model checkpointing and versioning
- Production-ready inference pipeline

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd price-optimizer
```

2. Install dependencies:
```bash
uv pip install torch pandas numpy gymnasium pytorch-forecasting wandb scikit-learn
```

## Project Structure

```
src/
├── price_optimizer/
│   ├── config/         # Configuration settings
│   │   └── config.py   # Modular configuration classes
│   ├── data/           # Data preprocessing
│   │   └── preprocessor.py  # Robust data handling
│   ├── env/            # RL environment
│   │   └── price_env.py    # Custom Gym environment
│   ├── models/         # Neural networks and agent
│   │   ├── networks.py     # Actor-Critic architectures
│   │   └── sac_agent.py    # SAC implementation
│   ├── train.py        # Training pipeline
│   └── inference.py    # Inference pipeline
└── main.py            # Main entry point
```

## Technical Architecture

### Data Pipeline
1. Data Loading & Validation
   - Required columns validation
   - Missing value handling
   - Data type verification
   - Chronological sorting

2. Feature Engineering
   - Moving averages (7-day window)
   - Time features (day of week, month)
   - Sales predictions
   - Conversion rate processing

3. Sequence Creation
   - Historical window generation
   - Feature normalization
   - State validation

### Environment Design
1. State Space
   - Historical prices (7 days)
   - Sales volumes
   - Conversion rates (organic & ad)
   - Profit metrics
   - Time features

2. Action Space
   - Continuous price range [-1, 1]
   - Scaled to actual prices
   - Bounded by min/max limits

3. Reward Function
   ```python
   reward = (
       0.4 * sales_reward +
       0.3 * price_reward +
       0.15 * organic_conversion_reward +
       0.15 * ad_conversion_reward +
       exploration_bonus -
       sales_drop_penalty
   )
   ```

### Neural Networks
1. Actor Network
   - Input normalization
   - Residual connections
   - Twin output heads (mean, log_std)
   - Tanh action squashing
   - Stability measures

2. Critic Network
   - Twin Q-networks
   - State-action input
   - Value bounds
   - Gradient clipping
   - Dropout regularization

## Usage

### Training

Basic training:
```bash
python src/main.py --mode train --seed 42
```

Advanced training options:
```bash
# Custom configuration
python src/main.py --mode train --seed 42 --config custom_config.py

# Resume from checkpoint
python src/main.py --mode train --seed 42 --resume checkpoint.pt
```

### Inference

Basic inference:
```bash
python src/main.py --mode inference --checkpoint path/to/checkpoint.pt
```

Advanced inference:
```bash
# Custom prediction horizon
python src/main.py --mode inference --checkpoint model.pt --days 14

# Batch predictions
python src/main.py --mode inference --checkpoint model.pt --batch-file products.csv
```

## Configuration

### Data Configuration
```python
@dataclass
class DataConfig:
    data_path: str = "data/woolballhistory.csv"
    history_window: int = 7
    train_test_split: float = 0.8
    features: List[str] = [
        "Product Price",
        "Organic Conversion Percentage",
        "Ad Conversion Percentage",
        "Total Profit",
        "Total Sales",
        "Predicted Sales"
    ]
```

### Environment Configuration
```python
@dataclass
class EnvConfig:
    min_price: float = 20.0
    max_price: float = 30.0
    price_step: float = 0.05
    exploration_bonus_scale: float = 0.3
    reward_weights: Dict[str, float] = {
        "sales": 0.4,
        "price": 0.3,
        "organic_conversion": 0.15,
        "ad_conversion": 0.15
    }
```

### Model Configuration
```python
@dataclass
class ModelConfig:
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
```

## Data Requirements

Required CSV format:
```csv
Report Date,Product Price,Organic Conversion Percentage,Ad Conversion Percentage,Total Profit,Total Sales,Predicted Sales
2024-01-01,25.99,2.5,1.8,1250.00,48,50
```

Field specifications:
- Report Date: YYYY-MM-DD format
- Product Price: Decimal number
- Conversion Percentages: 0-100 range
- Total Profit: Decimal number
- Total Sales: Integer
- Predicted Sales: Integer/Float

## Monitoring

### Training Metrics
- Actor/Critic losses
- Alpha value
- Reward components
- State statistics
- Action distributions

### Evaluation Metrics
- Average rewards
- Price statistics
- Sales performance
- Conversion rates
- Exploration metrics

## Best Practices

### Data Preparation
1. Data Quality
   - Clean historical data
   - Remove outliers
   - Handle missing values
   - Validate price ranges

2. Feature Engineering
   - Calculate moving averages
   - Add time features
   - Normalize values
   - Create sequences

### Training
1. Initial Setup
   - Start with default hyperparameters
   - Use small learning rates
   - Enable gradient clipping
   - Monitor reward components

2. Optimization
   - Adjust reward weights
   - Tune exploration parameters
   - Optimize network architecture
   - Balance batch sizes

### Production Deployment
1. Model Serving
   - Regular evaluation
   - Performance monitoring
   - Error handling
   - Fallback strategies

2. Maintenance
   - Regular retraining
   - Data validation
   - Model versioning
   - Performance tracking

## Troubleshooting

### Common Issues
1. Training Instability
   - Reduce learning rates
   - Increase batch size
   - Check reward scaling
   - Validate state normalization

2. Poor Performance
   - Verify data quality
   - Check reward weights
   - Adjust exploration parameters
   - Increase training duration

3. Production Issues
   - Validate input data
   - Check model loading
   - Monitor resource usage
   - Enable error logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## License

MIT License - see LICENSE file for details