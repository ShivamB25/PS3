"""Price optimization environment for reinforcement learning."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from gymnasium import spaces


class PriceOptimizationEnv(gym.Env):
    """Environment for price optimization using reinforcement learning."""

    def __init__(self, data_preprocessor, config):
        """Initialize the price optimization environment.
        
        Args:
            data_preprocessor: DataPreprocessor instance for handling data
            config: Configuration object containing environment parameters
        """
        super().__init__()

        self.preprocessor = data_preprocessor
        self.config = config
        self.current_step = 0
        self.max_steps = len(self.preprocessor.processed_data) - 1
        self.test_mode = False

        # Calculate price bounds from data
        prices = [d['Product Price'] for d in self.preprocessor.processed_data if 'Product Price' in d]
        self.min_price = float(min(prices)) * 0.8  # Allow 20% below minimum historical
        self.max_price = float(max(prices)) * 1.2  # Allow 20% above maximum historical
        self.price_range = self.max_price - self.min_price

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,  # Will be scaled to price range
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # State space includes all features from config
        self.state_dim = len(self.config.data.features)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional configuration including test_mode
            
        Returns:
            Initial state observation and info dict
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        if options and 'test_mode' in options:
            self.test_mode = options['test_mode']
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Price adjustment action (-1 to 1)
            
        Returns:
            tuple containing:
            - next state observation
            - reward
            - terminated flag
            - truncated flag
            - info dictionary
        """
        # Convert normalized action to price
        price = self.min_price + ((action[0] + 1) / 2) * self.price_range
        
        # Get current data
        current_data = self.preprocessor.processed_data[self.current_step]
        
        # Calculate metrics
        sales = self._estimate_demand(price, current_data)
        organic_conv = self._estimate_organic_conversion(price, current_data)
        ad_conv = self._estimate_ad_conversion(price, current_data)
        profit = self._calculate_profit(price, sales, organic_conv, ad_conv)
        
        # Calculate reward using weighted components
        reward = (
            self.config.env.reward_weights['sales'] * sales / current_data['Total Sales'] +
            self.config.env.reward_weights['price'] * price / current_data['Product Price'] +
            self.config.env.reward_weights['organic_conversion'] * organic_conv / current_data['Organic Conversion Percentage'] +
            self.config.env.reward_weights['ad_conversion'] * ad_conv / current_data['Ad Conversion Percentage']
        )
        
        # Add exploration bonus in training mode
        if not self.test_mode:
            exploration_bonus = self._calculate_exploration_bonus(price)
            reward += self.config.env.exploration_bonus_scale * exploration_bonus
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next state and info
        next_state = self._get_state()
        info = {
            'price': float(price),
            'sales': float(sales),
            'organic_conv': float(organic_conv),
            'ad_conv': float(ad_conv),
            'profit': float(profit)
        }
        
        return next_state, float(reward), terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        if self.current_step >= len(self.preprocessor.processed_data):
            return np.zeros(self.state_dim)
            
        current_data = self.preprocessor.processed_data[self.current_step]
        state = []
        
        for feature in self.config.data.features:
            if feature in current_data:
                state.append(current_data[feature])
            else:
                state.append(0.0)
        
        return np.array(state, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get current environment info."""
        current_data = self.preprocessor.processed_data[self.current_step]
        return {
            'price': float(current_data['Product Price']),
            'sales': float(current_data['Total Sales']),
            'organic_conv': float(current_data['Organic Conversion Percentage']),
            'ad_conv': float(current_data['Ad Conversion Percentage'])
        }

    def _estimate_demand(self, price: float, current_data: Dict[str, float]) -> float:
        """Estimate demand for a given price."""
        base_demand = current_data['Total Sales']
        price_elasticity = -1.5  # Price elasticity of demand
        
        price_change = (price - current_data['Product Price']) / current_data['Product Price']
        demand_change = price_elasticity * price_change
        
        # Apply temporal factors
        day_factor = 1.0 + 0.1 * np.cos(2 * np.pi * current_data['DayOfWeek'] / 7)  # Weekend effect
        month_factor = 1.0 + 0.05 * np.sin(2 * np.pi * current_data['Month'] / 12)  # Seasonal effect
        
        estimated_demand = base_demand * (1 + demand_change) * day_factor * month_factor
        return max(0, estimated_demand)

    def _estimate_organic_conversion(self, price: float, current_data: Dict[str, float]) -> float:
        """Estimate organic conversion rate."""
        base_conv = current_data['Organic Conversion Percentage']
        price_sensitivity = -0.5  # Conversion rate sensitivity to price
        
        price_change = (price - current_data['Product Price']) / current_data['Product Price']
        conv_change = price_sensitivity * price_change
        
        estimated_conv = base_conv * (1 + conv_change)
        return max(0, min(100, estimated_conv))  # Clip to valid percentage

    def _estimate_ad_conversion(self, price: float, current_data: Dict[str, float]) -> float:
        """Estimate ad conversion rate."""
        base_conv = current_data['Ad Conversion Percentage']
        price_sensitivity = -0.7  # Ad conversion more sensitive to price
        
        price_change = (price - current_data['Product Price']) / current_data['Product Price']
        conv_change = price_sensitivity * price_change
        
        estimated_conv = base_conv * (1 + conv_change)
        return max(0, min(100, estimated_conv))  # Clip to valid percentage

    def _calculate_profit(self, price: float, sales: float, organic_conv: float, ad_conv: float) -> float:
        """Calculate total profit."""
        # Simplified profit calculation
        revenue = price * sales
        cost_per_unit = price * 0.6  # Assume 60% cost
        marketing_cost = sales * (ad_conv / 100) * 0.1 * price  # 10% of price for converted ad sales
        
        total_cost = (cost_per_unit * sales) + marketing_cost
        return revenue - total_cost

    def _calculate_exploration_bonus(self, price: float) -> float:
        """Calculate exploration bonus based on price history."""
        price_history = [d['Product Price'] for d in self.preprocessor.processed_data[:self.current_step]]
        if not price_history:
            return 1.0
            
        # Calculate distance to nearest historical price
        min_distance = min(abs(price - p) for p in price_history)
        normalized_distance = min_distance / self.price_range
        
        return np.exp(-5 * normalized_distance)  # Exponential decay
