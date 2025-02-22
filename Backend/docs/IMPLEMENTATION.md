# Price Optimization System Implementation

## System Overview

The price optimization system uses Soft Actor-Critic (SAC) reinforcement learning to find optimal product pricing strategies. The system learns from historical data while balancing exploration of new price points with exploitation of known successful prices.

**System Architecture Diagram**

(Description: A high-level block diagram showing the main components of the system and their interactions. The diagram should include the following components:

1.  Data Source: Historical price and sales data (e.g., CSV file)
2.  Data Preprocessor: Handles data loading, cleaning, and normalization
3.  RL Environment: Simulates the price optimization problem
4.  SAC Agent: Learns the optimal pricing policy
5.  Training Loop: Iteratively trains the agent using the environment
6.  Evaluation Module: Evaluates the agent's performance
7.  Deployment: Integrates the trained agent into a real-world system)

**Data Flow Diagram**

(Description: A detailed data flow diagram showing how data moves through the system. The diagram should include the following steps:

1.  Load historical data from the data source
2.  Preprocess the data (handle missing values, normalize features)
3.  Create training sequences
4.  Initialize the RL environment with the preprocessed data
5.  The agent interacts with the environment by selecting actions (price adjustments)
6.  The environment returns the next state and reward
7.  The agent stores the transition in the replay buffer
8.  The agent samples a batch from the replay buffer
9.  The agent updates the actor and critic networks
10. The agent evaluates the performance and saves the best model)

## Code Structure

```
src/price_optimizer/
├── data/
│   └── preprocessor.py      # Data preprocessing, normalization, and environment implementation
├── models/
│   ├── networks.py         # Neural network architectures
│   ├── sac_agent.py        # SAC agent implementation
│   └── price_memory.py     # Price exploration memory system
├── config/
│   └── config.py           # Configuration parameters
└── train.py                # Training loop implementation
```

## Data Flow

1. **Data Loading & Preprocessing** (`preprocessor.py`):
   - Loads historical price and sales data
   - Handles missing values with forward/backward filling
   - Normalizes features to [0,1] range
   - Computes important statistics (median, max values)
   - Creates sequences for training
   - Validates data quality with warnings
   - Provides raw price history for exploration guidance

2. **Environment** (`price_env.py`):
   - Implements Gymnasium interface
   - Manages state space (price, sales, conversion rates)
   - Handles action space (price adjustments)
   - Calculates rewards based on multiple factors
   - Predicts next states using historical or synthetic data

3. **Agent** (`sac_agent.py`):
   - Implements SAC algorithm with automatic entropy tuning
   - Manages actor and critic networks
   - Handles experience replay buffer
   - Performs gradient-based updates
   - Tracks training metrics
   - Integrates with PriceMemory for smart exploration

## Key Features

### 1. Data Preprocessing
- **Missing Value Handling**:
  - Forward fills missing prices to maintain temporal patterns
  - Uses median for conversion rates
  - Zeros for missing profits/sales
  - Maintains null predicted sales for future dates

```python
# Example of handling missing values in preprocessor.py
df['Product Price'] = df['Product Price'].ffill().bfill()
df['Organic Conversion Percentage'] = df['Organic Conversion Percentage'].fillna(df['Organic Conversion Percentage'].median())
```

- **Data Validation**:
  - Checks for negative values in profits/sales
  - Validates conversion rate ranges (0-100%)
  - Reports data quality issues through warnings

### 2. Environment Design

#### State Space
- Product Price (normalized)
- Organic Conversion Rate
- Ad Conversion Rate
- Total Profit
- Total Sales
- Predicted Sales

#### Action Space
- Continuous price adjustments in [-1, 1]
- Scaled to actual price range
- Allows exploration up to 20% above historical maximum

#### Reward Function
```python
# Reward calculation in price_env.py
total_reward = (
    0.6 * sales_reward +      # Primary focus on sales
    0.2 * price_reward +      # Encourage price exploration
    0.2 * conv_reward         # Reward conversion improvements
)
```

- **Sales Reward**:
  - 2x bonus for sales increase
  - 4x penalty for sales decrease
  - Normalized by maximum historical sales

- **Price Exploration Reward**:
  - Rewards testing higher prices
  - Only given if sales maintain within 90% of current
  - Scales with distance from median price

- **Conversion Rate Reward**:
  - Bonus for improving organic and ad conversion rates
  - Scaled relative to median conversion rates

### 3. Exploration vs Exploitation

#### Smart Exploration System (PriceMemory)
1. **Dynamic Price Tracking**:
   - Maintains history of explored price points
   - Tracks success metrics for different price regions
   - Uses exponential moving average for reward history
   - Automatically adapts bin sizes based on data distribution

2. **Adaptive Exploration**:
   - Dynamically adjusts exploration rate based on performance
   - Higher exploration for unexplored regions
   - Gradually reduces exploration in well-understood areas
   - Uses decay factor to allow revisiting previously explored regions

3. **Guided Exploration**:
   - Calculates optimal exploration std based on price distribution
   - Provides exploration bonuses for unexplored regions
   - Combines network uncertainty with exploration guidance
   - Adapts exploration parameters based on data statistics

#### Exploitation Mechanisms
1. **Historical Data Usage**:
   - Finds similar historical price points
   - Weighted averaging based on price similarity
   - Preserves successful price-sales relationships
   - Tracks promising price regions based on success metrics

2. **Performance Incentives**:
   - Strong rewards for sales improvements
   - Penalties for underperforming median sales
   - Considers conversion rate improvements
   - Incorporates exploration bonuses into reward calculation

### 4. Training Process

1. **Episode Structure**:
   - Starts from random historical state
   - Runs for configured number of steps
   - Updates based on accumulated experience
   - Maintains exploration memory across episodes

2. **Learning Updates**:
   - Batched updates from replay buffer
   - Soft updates to target networks
   - Automatic entropy adjustment
   - Exploration-aware reward augmentation

3. **Monitoring**:
   - Tracks actor and critic losses
   - Monitors gradient norms
   - Records price and sales statistics
   - Regular evaluation episodes
   - Tracks exploration metrics

### 5. Inference Process

1. **Minimal Exploration**:
   - Uses reduced exploration rate (0.1) during inference
   - Maintains ability to adapt to new situations
   - Focuses on exploiting learned strategies

2. **Temporal Adaptation**:
   - Adjusts prices based on day of week
   - Considers seasonal patterns
   - Uses reduced adjustment factors for stability

3. **Price Adjustments**:
   - Applies temporal factors to base predictions
   - Ensures prices stay within reasonable bounds
   - Maintains consistency with training behavior

## System Setup and Usage

### 1. Backend Setup

First, ensure all Python dependencies are installed using uv:
```bash
uv pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn src.price_optimizer.api:app --reload --port 8000
```

The API will be available at:
- API Endpoints: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc Documentation: http://localhost:8000/redoc

### 2. Frontend Setup

Install frontend dependencies:
```bash
cd frontend
bun install
```

Start the Next.js development server:
```bash
bun run dev --port 3001
```

The frontend will be available at:
- Web Interface: http://localhost:3001

### 3. Running the Complete System

1. **Start the Backend Server**
   ```bash
   # Terminal 1
   uvicorn src.price_optimizer.api:app --reload --port 8000
   ```

2. **Start the Frontend Development Server**
   ```bash
   # Terminal 2
   cd frontend
   bun run dev
   ```

3. **Access the System**
   - Backend API: http://localhost:8000
   - Frontend Interface: http://localhost:3001
   - API Documentation: http://localhost:8000/docs

### 4. Training

There are two ways to train models:

a. Using the CLI:
```bash
PYTHONPATH=. python3 src/main.py --mode train
```

b. Using the API:
```bash
curl -X POST "http://localhost:8000/train/woolball" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/woolballhistory.csv"
```

### 5. Inference

a. Using the CLI:
```bash
python src/main.py --mode inference --checkpoint checkpoints/best_model.pt --data-path data/history.csv
```

b. Using the API:
```bash
curl -X POST "http://localhost:8000/predict/woolball" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/woolballhistory.csv" \
  -F 'config={
    "future_dates": ["2025-01-23", "2025-01-24"],
    "exploration_mode": false
  }'
```

### 6. Visualizations

Generate visualizations through the API:
```bash
# Historical Analysis
curl -X POST "http://localhost:8000/viz/visualize/historical/woolball" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/woolballhistory.csv"

# Prediction Analysis
curl -X POST "http://localhost:8000/viz/visualize/predictions/woolball" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/woolballhistory.csv" \
  -F 'predictions=[...]'

# Model Performance
curl "http://localhost:8000/viz/visualize/model/woolball"
```

### 7. Monitoring

- Training progress tracked in WandB
- Key metrics: rewards, prices, sales
- Gradient and loss monitoring
- Exploration statistics tracking
- API endpoints for visualization and analysis

## Future Improvements

1. **Data Augmentation**:
   - Implement more sophisticated synthetic data generation
   - Better handling of seasonal patterns
   - Enhanced price elasticity modeling

2. **Advanced Exploration**:
   - Implement curiosity-driven exploration
   - Add uncertainty estimation
   - Dynamic entropy adjustment
   - Enhanced price region discovery

3. **Performance Optimization**:
   - Batch processing of similar states
   - Improved experience replay sampling
   - Enhanced reward scaling
   - More sophisticated temporal adaptation

## Glossary

- **SAC**: Soft Actor-Critic, an off-policy reinforcement learning algorithm that aims to maximize both reward and entropy.
- **Exploration**: The process of trying new actions or states to discover potentially better strategies.
- **Exploitation**: The process of using known successful strategies to maximize reward.
- **Replay Buffer**: A memory buffer that stores past experiences (state, action, reward, next state) for training the agent.
- **Gradient Clipping**: A technique used to prevent exploding gradients during training by limiting the maximum value of the gradients.
- **PriceMemory**: A system that tracks explored price points and guides exploration strategy based on historical performance.
- **Temporal Adaptation**: The process of adjusting prices based on time-related factors such as day of week and seasonality.