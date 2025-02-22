"""Inference script for making price predictions."""

import torch
import numpy as np
from typing import Dict, List
import pandas as pd
from datetime import datetime, timedelta

from .config.config import Config
from .data.preprocessor import DataPreprocessor
from .env.price_env import PriceOptimizationEnv
from .models.sac_agent import SACAgent


class PricePredictor:
    """Class for making price predictions using trained model."""

    def __init__(self, config: Config, checkpoint_path: str):
        """Initialize the predictor.

        Args:
            config: System configuration
            checkpoint_path: Path to trained model checkpoint
        """
        self.config = config

        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(config.data)
        self.preprocessor.prepare_data()

        # Initialize environment
        self.env = PriceOptimizationEnv(self.preprocessor, config.env)

        # Initialize agent with price history
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Get price history for initialization
        price_history = self.preprocessor.get_price_history()

        # Initialize agent with price history but minimal exploration for inference
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config.model,
            price_history=price_history,
        )

        # Load trained weights
        self.agent.load(checkpoint_path)

        # Set exploration parameters for inference
        self.agent.price_memory.exploration_rate = (
            0.1  # Minimal exploration during inference
        )

    def predict_price(self, current_state: np.ndarray) -> Dict[str, float]:
        """Predict optimal price for given state.

        Args:
            current_state: Current state of the environment

        Returns:
            Dictionary containing prediction details
        """
        # Get base action with minimal exploration
        action = self.agent.select_action(current_state, evaluate=True)

        # Get temporal context
        state_reshaped = current_state.reshape(
            -1, len(self.preprocessor.config.features)
        )
        current_day = int(state_reshaped[-1][8] * 7)  # DayOfWeek
        current_month = int(state_reshaped[-1][9] * 12) + 1  # Month

        # Calculate temporal adjustment factors
        day_factor = 1.0 + 0.05 * np.cos(
            2 * np.pi * current_day / 7
        )  # Reduced weekend adjustment
        month_factor = 1.0 + 0.025 * np.sin(
            2 * np.pi * current_month / 12
        )  # Reduced seasonal variation

        # Get base price and apply temporal adjustments
        base_price = self.env.min_price + ((action[0] + 1) / 2) * self.env.price_range
        adjusted_price = base_price * day_factor * month_factor

        # Convert back to normalized action
        adjusted_action = action.copy()
        adjusted_action[0] = (
            2 * (adjusted_price - self.env.min_price) / self.env.price_range
        ) - 1
        adjusted_action = np.clip(adjusted_action, -1.0, 1.0)

        # Simulate step with adjusted action
        next_state, reward, _, _, info = self.env.step(adjusted_action)

        return {
            "recommended_price": float(adjusted_price),
            "predicted_sales": float(info["sales"]),
            "predicted_organic_conv": float(info["organic_conv"]),
            "predicted_ad_conv": float(info["ad_conv"]),
            "predicted_reward": float(reward),
        }

    def predict_price_sequence(self, days: int = 7) -> List[Dict[str, float]]:
        """Predict sequence of optimal prices for multiple days.

        Args:
            days: Number of days to predict for

        Returns:
            List of prediction dictionaries
        """
        predictions = []
        state, _ = self.env.reset()

        for _ in range(days):
            # Get base action with minimal exploration
            action = self.agent.select_action(state, evaluate=True)

            # Get temporal context
            state_reshaped = state.reshape(-1, len(self.preprocessor.config.features))
            current_day = int(state_reshaped[-1][8] * 7)  # DayOfWeek
            current_month = int(state_reshaped[-1][9] * 12) + 1  # Month

            # Calculate temporal adjustment factors
            day_factor = 1.0 + 0.05 * np.cos(
                2 * np.pi * current_day / 7
            )  # Reduced weekend adjustment
            month_factor = 1.0 + 0.025 * np.sin(
                2 * np.pi * current_month / 12
            )  # Reduced seasonal variation

            # Get price from action
            base_price = (
                self.env.min_price + ((action[0] + 1) / 2) * self.env.price_range
            )

            # Apply temporal adjustments to price
            adjusted_price = base_price * day_factor * month_factor

            # Convert back to normalized action
            adjusted_action = action.copy()
            adjusted_action[0] = (
                2 * (adjusted_price - self.env.min_price) / self.env.price_range
            ) - 1
            adjusted_action = np.clip(adjusted_action, -1.0, 1.0)

            # Get prediction details using adjusted action
            next_state, reward, _, _, info = self.env.step(adjusted_action)

            # Store prediction with adjusted price
            predictions.append(
                {
                    "recommended_price": float(adjusted_price),
                    "predicted_sales": float(info["sales"]),
                    "predicted_organic_conv": float(info["organic_conv"]),
                    "predicted_ad_conv": float(info["ad_conv"]),
                    "predicted_reward": float(reward),
                }
            )

            # Update state
            state = next_state

        return predictions

    def generate_report(self, predictions: List[Dict[str, float]]) -> pd.DataFrame:
        """Generate a report from predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            DataFrame containing the report
        """
        dates = [
            (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(len(predictions))
        ]

        report = pd.DataFrame(predictions)
        report["date"] = dates
        report = report[
            [
                "date",
                "recommended_price",
                "predicted_sales",
                "predicted_organic_conv",
                "predicted_ad_conv",
                "predicted_reward",
            ]
        ]

        return report


def run_inference(config: Config, checkpoint_path: str):
    """Run inference using trained model.

    Args:
        config: System configuration
        checkpoint_path: Path to trained model checkpoint
    """
    # Initialize predictor
    predictor = PricePredictor(config, checkpoint_path)

    # Generate predictions
    predictions = predictor.predict_price_sequence(days=7)

    # Generate and save report
    report = predictor.generate_report(predictions)

    # Print predictions
    print("\nPrice Predictions for Next 7 Days:")
    print("==================================")
    for _, row in report.iterrows():
        print(f"\nDate: {row['date']}")
        print(f"Recommended Price: ${row['recommended_price']:.2f}")
        print(f"Predicted Sales: {row['predicted_sales']:.2f}")
        print(f"Predicted Organic Conversion: {row['predicted_organic_conv']:.2f}%")
        print(f"Predicted Ad Conversion: {row['predicted_ad_conv']:.2f}%")
        print(f"Predicted Reward: {row['predicted_reward']:.2f}")

    # Save report
    report_path = "price_predictions.csv"
    report.to_csv(report_path, index=False)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run price prediction inference")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    args = parser.parse_args()

    config = Config()
    run_inference(config, args.checkpoint)
