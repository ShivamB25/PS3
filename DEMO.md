# Price Optimization System Demo Guide

This guide outlines how to effectively demonstrate the price optimization system to judges.

## 1. Setup (Pre-Demo)

### 1.1 Environment Preparation
```bash
# Install dependencies
uv pip install torch pandas numpy gymnasium pytorch-forecasting wandb scikit-learn

# Clone and setup
git clone <repository-url>
cd price-optimizer
```

### 1.2 Data Preparation
1. Have both datasets ready:
   - woolballhistory.csv
   - soapnutshistory.csv

2. Prepare visualizations of historical data:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load and plot historical data
df = pd.read_csv("data/woolballhistory.csv")
plt.figure(figsize=(10, 6))
plt.plot(df['Product Price'], label='Historical Prices')
plt.axhline(y=df['Product Price'].median(), color='r', linestyle='--', label='Median Price')
plt.title('Historical Price Distribution')
plt.show()
```

## 2. Live Demo Flow

### 2.1 Problem Introduction (2 minutes)
1. Show the challenge:
   - Price optimization problem
   - Historical price median barrier
   - Need for intelligent exploration

2. Display historical data visualization:
   - Price distribution
   - Sales correlation
   - Current median price

### 2.2 Solution Architecture (3 minutes)
1. Show system components:
```
Data Pipeline → RL Environment → SAC Agent → Price Predictions
```

2. Highlight key features:
   - Continuous price optimization
   - Multi-objective rewards
   - Exploration mechanism
   - Safety constraints

### 2.3 Training Demonstration (5 minutes)

1. Start training with Weights & Biases visualization:
```bash
# Run training with wandb logging
python src/main.py --mode train --seed 42
```

2. Show in Weights & Biases:
   - Training metrics
   - Reward components
   - Price evolution
   - Sales performance

3. Key points to highlight:
   - Price progression above median
   - Sales maintenance
   - Conversion rate stability
   - Exploration behavior

### 2.4 Live Inference (5 minutes)

1. Load trained model:
```bash
# Run inference
python src/main.py --mode inference --checkpoint checkpoints/best_model.pt
```

2. Show price recommendations:
   - Next day predictions
   - 7-day forecast
   - Expected sales impact
   - Conversion projections

3. Demonstrate with different scenarios:
```python
# Example scenarios to show
scenarios = [
    {"initial_price": 14.00, "expected": "~14.80"},
    {"initial_price": 15.70, "expected": "~16.20"},
    {"initial_price": 16.20, "expected": "~16.60"}
]
```

### 2.5 Results Analysis (3 minutes)

1. Show key achievements:
   - Price progression example:
     ```
     $14.00 → $14.80 → $15.70 → $16.20 → $16.60
     ```
   - Sales maintenance
   - Conversion stability

2. Display performance metrics:
   - Average price increase
   - Sales impact
   - Conversion rates
   - Exploration statistics

## 3. Interactive Demonstration

### 3.1 Price Prediction Demo
```python
from price_optimizer.inference import PricePredictor
predictor = PricePredictor(config, "checkpoints/best_model.pt")

# Live price prediction
current_price = float(input("Enter current price: "))
prediction = predictor.predict_price(current_price)
print(f"Recommended price: ${prediction['recommended_price']:.2f}")
print(f"Predicted sales: {prediction['predicted_sales']:.0f}")
```

### 3.2 What-If Analysis
```python
# Show different price points
prices = [14.0, 15.0, 16.0, 17.0]
for price in prices:
    prediction = predictor.predict_price(price)
    print(f"\nInput price: ${price:.2f}")
    print(f"Recommended: ${prediction['recommended_price']:.2f}")
    print(f"Expected sales: {prediction['predicted_sales']:.0f}")
```

## 4. Technical Deep Dive (If Asked)

### 4.1 Reward Function
```python
# Show reward calculation
reward = (
    0.4 * sales_reward +          # Sales performance
    0.3 * price_reward +          # Price optimization
    0.15 * organic_conv_reward +  # Organic conversion
    0.15 * ad_conv_reward +       # Ad conversion
    exploration_bonus            # Higher price exploration
)
```

### 4.2 Exploration Mechanism
```python
# Show exploration bonus calculation
if norm_price > norm_median_price:
    exploration_bonus = exploration_bonus_scale * (norm_price - norm_median_price)
```

## 5. Backup Plans

### 5.1 Alternative Scenarios
- Have multiple trained models ready
- Prepare different data scenarios
- Have offline visualizations ready

### 5.2 Common Questions
1. How does it handle market changes?
   - Show adaptive behavior
   - Demonstrate safety constraints

2. What prevents price collapse?
   - Show reward components
   - Demonstrate stability measures

3. How is exploration balanced?
   - Show entropy adjustment
   - Demonstrate progressive exploration

## 6. Demo Checklist

### 6.1 Pre-Demo
- [ ] Test all dependencies
- [ ] Prepare both datasets
- [ ] Train and save models
- [ ] Setup Weights & Biases
- [ ] Prepare visualizations
- [ ] Test inference scripts

### 6.2 During Demo
- [ ] Monitor system resources
- [ ] Keep terminal outputs clear
- [ ] Have code examples ready
- [ ] Track timing for each section

### 6.3 Post-Demo
- [ ] Save generated predictions
- [ ] Export visualizations
- [ ] Document any questions
- [ ] Note improvement suggestions

## 7. Key Talking Points

1. Technical Innovation
   - SAC algorithm adaptation
   - Custom reward design
   - Stability measures

2. Business Impact
   - Progressive price optimization
   - Sales maintenance
   - Risk management

3. Practical Implementation
   - Production readiness
   - Monitoring capabilities
   - Safety constraints

4. Future Potential
   - Multi-product optimization
   - Market adaptation
   - Enhanced exploration

Remember:
- Keep the demo focused and concise
- Show practical results first
- Have technical details ready
- Be prepared for questions
- Demonstrate live functionality
- Highlight business value