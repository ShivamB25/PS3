"""Price optimization environment implementing the Gymnasium interface."""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from gymnasium import spaces
from ..config.config import EnvConfig
from ..data.preprocessor import DataPreprocessor

class PriceOptimizationEnv(gym.Env):
    """Custom Environment for price optimization."""
    
    def __init__(self, preprocessor: DataPreprocessor, config: EnvConfig):
        """Initialize the environment."""
        super().__init__()
        
        self.preprocessor = preprocessor
        self.config = config
        self.stats = preprocessor.stats
        
        # Set price bounds for gradual exploration
        self.min_price = self.stats['min_price']
        self.max_price = self.stats['max_price'] * 1.1  # Allow 10% above max historical
        self.price_range = self.max_price - self.min_price
        self.median_price = self.stats['median_price']
        self.optimal_price = self.median_price  # Track discovered optimal price
        self.price_step = self.price_range / 200  # Smaller steps for gradual changes
        
        # Track performance metrics
        self.max_historical_sales = self.stats['max_sales']
        self.median_sales = self.stats['median_sales']
        self.median_organic_conv = self.stats['median_organic_conv']
        self.median_ad_conv = self.stats['median_ad_conv']
        
        # Price optimization parameters
        self.price_momentum = 0.0  # Track direction of successful price changes
        self.success_threshold = 0.95  # Sales ratio threshold for success
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        num_features = len(preprocessor.config.features)
        history_window = preprocessor.config.history_window
        self.state_dim = history_window * num_features
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.current_state = None
        self.episode_history = []
        
        # Dynamic exploration parameters
        self.exploration_threshold = self.median_price
        self.max_price_step = (self.max_price - self.median_price) / 20  # Gradual price increase
        
    def _calculate_reward(
        self,
        price: float,
        current_price: float,
        next_state: np.ndarray,
        current_sales: float,
        current_organic_conv: float,
        current_ad_conv: float
    ) -> float:
        """Calculate reward with dynamic scaling based on historical performance."""
        # Sales performance (primary objective)
        predicted_sales = next_state[4] * (self.stats['max_sales'] - self.stats['min_sales']) + self.stats['min_sales']
        sales_change = (predicted_sales - current_sales) / (self.max_historical_sales + 1e-8)
        
        # Base reward from sales with normalized scaling
        sales_change = np.clip(sales_change, -1.0, 1.0)  # Normalize change
        if predicted_sales > current_sales:
            sales_reward = 1.0 * sales_change  # Bonus for sales increase
        else:
            sales_reward = 2.0 * sales_change  # Moderate penalty for sales decrease
        
        # Get current state metrics
        state_reshaped = self.current_state.reshape(-1, len(self.preprocessor.config.features))
        current_state = state_reshaped[-1]
        
        # Get temporal patterns for reward calculation
        current_day = int(current_state[8] * 7)  # Denormalize DayOfWeek
        current_month = int(current_state[9] * 12) + 1  # Denormalize Month
        
        # Calculate expected price based on temporal patterns
        day_factor = 1.0 + 0.1 * np.cos(2 * np.pi * current_day / 7)  # Higher prices on weekends
        month_factor = 1.0 + 0.05 * np.sin(2 * np.pi * current_month / 12)  # Seasonal variation
        expected_price = self.median_price * day_factor * month_factor
        
        # Price increase reward
        price_change = (price - self.median_price) / self.median_price
        sales_ratio = predicted_sales / current_sales if current_sales > 0 else 0.0
        
        # Price reward with normalized scaling
        price_change = np.clip(price_change, -1.0, 1.0)  # Normalize price change
        if price > self.median_price:
            if sales_ratio >= 0.95:  # Allow 5% sales drop for higher prices
                price_reward = 0.3 * price_change  # Smaller reward for price increase
            elif sales_ratio >= 0.9:  # Smaller reward for moderate sales drop
                price_reward = 0.1 * price_change
            else:
                price_reward = -0.2 * price_change  # Penalty for significant sales drop
        else:
            price_reward = -0.1 * abs(price_change)  # Small penalty for lower prices
        
        # Conversion rate rewards with normalized scaling
        organic_conv = next_state[1] * 100  # Denormalize
        ad_conv = next_state[2] * 100  # Denormalize
        
        # Normalize conversion improvements
        conv_reward = 0
        if organic_conv > self.median_organic_conv:
            organic_improvement = (organic_conv - self.median_organic_conv) / (self.median_organic_conv + 1e-8)
            conv_reward += 0.1 * np.clip(organic_improvement, 0, 1.0)
        if ad_conv > self.median_ad_conv:
            ad_improvement = (ad_conv - self.median_ad_conv) / (self.median_ad_conv + 1e-8)
            conv_reward += 0.1 * np.clip(ad_improvement, 0, 1.0)
        
        # Calculate optimal price progress
        optimal_progress = (self.optimal_price - self.median_price) / self.median_price
        optimal_factor = np.clip(optimal_progress, 0, 0.5)  # Cap at 50% improvement
        
        # Fixed reward weighting for stability
        sales_weight = 0.4  # Primary focus on sales
        price_weight = 0.4  # Equal focus on price optimization
        conv_weight = 0.2  # Secondary focus on conversions
        
        # Combine rewards with fixed weighting
        total_reward = (
            sales_weight * sales_reward +
            price_weight * price_reward +
            conv_weight * conv_reward
        )
        
        # Add smaller penalty for severe sales drop
        if predicted_sales < self.stats['median_sales'] * 0.8:
            total_reward -= 0.5  # Reduced penalty
        
        # Normalize reward components
        total_reward = np.clip(total_reward, -1.0, 1.0)  # Base normalization
        
        # Add smaller stability bonus
        if predicted_sales >= current_sales * 0.95:
            price_diff_ratio = abs(price - self.optimal_price) / self.price_step
            stability_bonus = 0.1 * np.exp(-price_diff_ratio)  # Smaller bonus with exponential decay
            total_reward += stability_bonus
        
        # Final scaling with smaller factor
        reward_scale = 2.0  # Reduced scale for more reasonable values
        return float(total_reward * reward_scale)
    
    def _predict_next_state(self, price: float) -> np.ndarray:
        """Predict next state with emphasis on predicted sales and temporal patterns."""
        try:
            # Get current state metrics
            state_reshaped = self.current_state.reshape(-1, len(self.preprocessor.config.features))
            current_state = state_reshaped[-1]
            
            # Get current temporal features
            current_day = int(current_state[8] * 7)  # Denormalize DayOfWeek
            current_month = int(current_state[9] * 12) + 1  # Denormalize Month
            
            # Calculate next day's temporal features
            next_day = (current_day + 1) % 7
            next_month = current_month if next_day > current_day else (current_month % 12) + 1
            
            # Get valid historical data points for the next day of week
            df = self.preprocessor.data.copy()
            df = df.dropna(subset=['Product Price', 'Total Sales'])
            df_next_day = df[df['DayOfWeek'] == next_day]
            
            if len(df_next_day) == 0:
                return self._create_synthetic_state(price, next_day, next_month)
            
            # Find similar price and temporal patterns
            historical_prices = df_next_day['Product Price'].values
            price_diffs = np.abs(historical_prices - price)
            
            # Get temporal patterns
            day_factor = 1.0 + 0.1 * np.cos(2 * np.pi * next_day / 7)  # Higher prices on weekends
            month_factor = 1.0 + 0.05 * np.sin(2 * np.pi * next_month / 12)  # Seasonal variation
            
            # Get price trend with temporal adjustment
            current_ma7 = current_state[6] * (self.stats['max_price'] - self.stats['min_price']) + self.stats['min_price']
            expected_price = current_ma7 * day_factor * month_factor
            price_trend = (price - expected_price) / (expected_price + 1e-8)
            
            # Get sales trend
            current_sales_ma7 = current_state[7] * (self.stats['max_sales'] - self.stats['min_sales']) + self.stats['min_sales']
            
            # Create state with normalized values
            state = np.zeros(len(self.preprocessor.config.features))
            
            # Set price and moving average
            state[0] = self._safe_normalize(price, self.min_price, self.max_price)
            state[6] = self._safe_normalize(
                0.9 * price + 0.1 * current_ma7,  # Exponential moving average
                self.stats['min_price'],
                self.stats['max_price']
            )
            
            # Get current metrics from state
            current_price = current_state[0] * (self.stats['max_price'] - self.stats['min_price']) + self.stats['min_price']
            current_sales = current_state[4] * (self.stats['max_sales'] - self.stats['min_sales']) + self.stats['min_sales']

            # Use momentum to adjust predictions
            momentum_factor = 1.0 + 0.1 * self.price_momentum
            similar_day_sales = df_next_day['Total Sales'].mean()
            
            # Predict sales with momentum and gradual price adjustment
            price_sensitivity = 0.3  # Reduced from 0.5 for more gradual changes
            predicted_sales = similar_day_sales * (1.0 - price_sensitivity * price_trend) * momentum_factor
            predicted_sales = np.clip(predicted_sales, self.stats['min_sales'], self.stats['max_sales'])
            
            # Update price momentum based on success
            if predicted_sales >= current_sales * self.success_threshold:
                # Increase momentum for successful price changes
                price_diff = price - current_price
                if price_diff != 0:
                    self.price_momentum = 0.9 * self.price_momentum + 0.1 * np.sign(price_diff)
            else:
                # Reduce momentum for unsuccessful changes
                self.price_momentum *= 0.5
            
            # Update optimal price if we found a better point
            if predicted_sales >= current_sales * self.success_threshold and price > self.optimal_price:
                self.optimal_price = price
            
            # Set sales and moving average
            state[4] = self._safe_normalize(predicted_sales, self.stats['min_sales'], self.stats['max_sales'])
            state[7] = self._safe_normalize(
                0.9 * predicted_sales + 0.1 * current_sales_ma7,
                self.stats['min_sales'],
                self.stats['max_sales']
            )
            
            # Get conversion rates for similar days
            state[1] = self._safe_normalize(df_next_day['Organic Conversion Percentage'].mean(), 0, 100)
            state[2] = self._safe_normalize(df_next_day['Ad Conversion Percentage'].mean(), 0, 100)
            
            # Predict profit based on price and sales
            profit_margin = (price - self.median_price) / (self.median_price + 1e-8)
            predicted_profit = predicted_sales * price * (1.0 + profit_margin)
            state[3] = self._safe_normalize(predicted_profit, self.stats['min_profit'], self.stats['max_profit'])
            
            # Set predicted sales slightly different from actual sales
            state[5] = self._safe_normalize(
                predicted_sales * np.random.normal(1.0, 0.1),
                self.stats['min_sales'],
                self.stats['max_sales']
            )
            
            # Set temporal features
            state[8] = next_day / 7.0  # Normalize DayOfWeek
            state[9] = (next_month - 1) / 12.0  # Normalize Month
            
            return np.clip(state, 0, 1)
            
        except Exception as e:
            print(f"Error in state prediction: {str(e)}")
            return self._create_synthetic_state(price, (current_day + 1) % 7, next_month)
    
    def _create_synthetic_state(self, price: float, next_day: int, next_month: int) -> np.ndarray:
        """Create a synthetic state when no historical data is available."""
        state = np.zeros(len(self.preprocessor.config.features))
        
        # Get current state for trends if available
        if self.current_state is not None:
            state_reshaped = self.current_state.reshape(-1, len(self.preprocessor.config.features))
            current_state = state_reshaped[-1]
            # Get current metrics from state
            current_price = current_state[0] * (self.stats['max_price'] - self.stats['min_price']) + self.stats['min_price']
            current_sales = current_state[4] * (self.stats['max_sales'] - self.stats['min_sales']) + self.stats['min_sales']
            current_ma7 = current_state[6] * (self.stats['max_price'] - self.stats['min_price']) + self.stats['min_price']
            current_sales_ma7 = current_state[7] * (self.stats['max_sales'] - self.stats['min_sales']) + self.stats['min_sales']
        else:
            current_ma7 = self.median_price
            current_sales_ma7 = self.median_sales
        
        # Set price and moving average
        state[0] = self._safe_normalize(price, self.min_price, self.max_price)
        state[6] = self._safe_normalize(
            0.9 * price + 0.1 * current_ma7,  # Exponential moving average
            self.stats['min_price'],
            self.stats['max_price']
        )
        
        # Get temporal patterns
        day_factor = 1.0 + 0.1 * np.cos(2 * np.pi * next_day / 7)  # Higher prices on weekends
        month_factor = 1.0 + 0.05 * np.sin(2 * np.pi * next_month / 12)  # Seasonal variation
        
        # Calculate price trend with temporal adjustment
        expected_price = current_ma7 * day_factor * month_factor
        price_trend = (price - expected_price) / (expected_price + 1e-8)
        
        # Use momentum to adjust predictions
        momentum_factor = 1.0 + 0.1 * self.price_momentum
        base_sales = self.median_sales * day_factor * month_factor
        
        # Predict sales with momentum and gradual price adjustment
        price_sensitivity = 0.3  # Reduced from 0.5 for more gradual changes
        estimated_sales = base_sales * (1.0 - price_sensitivity * price_trend) * momentum_factor
        estimated_sales = np.clip(estimated_sales, self.stats['min_sales'], self.stats['max_sales'])
        
        # Update price momentum based on success
        if self.current_state is not None:
            if estimated_sales >= current_sales * self.success_threshold:
                price_diff = price - current_price
                if price_diff != 0:
                    self.price_momentum = 0.9 * self.price_momentum + 0.1 * np.sign(price_diff)
            else:
                self.price_momentum *= 0.5
        
        # Update optimal price if we found a better point
        if self.current_state is not None and estimated_sales >= current_sales * self.success_threshold and price > self.optimal_price:
            self.optimal_price = price
        
        # Set sales and moving average
        state[4] = self._safe_normalize(estimated_sales, self.stats['min_sales'], self.stats['max_sales'])
        state[7] = self._safe_normalize(
            0.9 * estimated_sales + 0.1 * current_sales_ma7,
            self.stats['min_sales'],
            self.stats['max_sales']
        )
        
        # Set conversion rates with temporal patterns
        base_organic = 0.5 + 0.1 * np.sin(2 * np.pi * next_day / 7)  # Daily pattern
        base_ad = 0.5 + 0.1 * np.cos(2 * np.pi * next_month / 12)  # Monthly pattern
        
        state[1] = np.clip(base_organic + np.random.normal(0, 0.05), 0, 1)  # Organic conversion
        state[2] = np.clip(base_ad + np.random.normal(0, 0.05), 0, 1)  # Ad conversion
        
        # Calculate profit
        profit_margin = (price - self.median_price) / (self.median_price + 1e-8)
        estimated_profit = estimated_sales * price * (1.0 + profit_margin)
        state[3] = self._safe_normalize(estimated_profit, self.stats['min_profit'], self.stats['max_profit'])
        
        # Set predicted sales
        state[5] = self._safe_normalize(
            estimated_sales * np.random.normal(1.0, 0.1),
            self.stats['min_sales'],
            self.stats['max_sales']
        )
        
        # Set temporal features
        state[8] = next_day / 7.0  # Normalize DayOfWeek
        state[9] = (next_month - 1) / 12.0  # Normalize Month
        
        return np.clip(state, 0, 1)
    
    def _safe_normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Safely normalize a value to [0,1] range."""
        if np.isnan(value) or min_val >= max_val:
            return 0.5
        return float(np.clip((value - min_val) / (max_val - min_val), 0, 1))
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.episode_history = []
        
        # Get initial state
        sequences = self.preprocessor.test_sequences if options and options.get('test_mode') else self.preprocessor.train_sequences
        if len(sequences) == 0:
            raise ValueError("No sequences available")
            
        start_idx = self.np_random.integers(0, len(sequences))
        self.current_state = sequences[start_idx]
        
        # Prepare info
        state_reshaped = self.current_state.reshape(-1, len(self.preprocessor.config.features))
        last_state = state_reshaped[-1]
        
        info = {
            'initial_price': float(last_state[0] * self.price_range + self.min_price),
            'initial_sales': float(last_state[4] * self.max_historical_sales),
            'initial_organic_conv': float(last_state[1] * 100),
            'initial_ad_conv': float(last_state[2] * 100)
        }
        
        return self.current_state.flatten(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Validate action
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to price
        normalized = (action[0] + 1) / 2
        price = self.min_price + normalized * self.price_range
        
        # Get current state metrics
        state_reshaped = self.current_state.reshape(-1, len(self.preprocessor.config.features))
        current_price = state_reshaped[-1][0] * self.price_range + self.min_price
        current_sales = state_reshaped[-1][4] * self.max_historical_sales
        current_organic_conv = state_reshaped[-1][1] * 100
        current_ad_conv = state_reshaped[-1][2] * 100
        
        # Predict next state
        next_state = self._predict_next_state(price)
        
        # Calculate reward
        reward = self._calculate_reward(
            price=price,
            current_price=current_price,
            next_state=next_state,
            current_sales=current_sales,
            current_organic_conv=current_organic_conv,
            current_ad_conv=current_ad_conv
        )
        
        # Update state
        self.current_state = np.roll(state_reshaped, -1, axis=0)
        self.current_state[-1] = next_state
        
        # Update step counter
        self.current_step += 1
        done = self.current_step >= self.preprocessor.config.history_window
        
        # Prepare info
        info = {
            'price': price,
            'sales': float(next_state[4] * self.max_historical_sales),
            'organic_conv': float(next_state[1] * 100),
            'ad_conv': float(next_state[2] * 100)
        }
        
        return self.current_state.flatten(), reward, done, False, info