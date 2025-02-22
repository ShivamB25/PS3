"""Data preprocessing for the price optimization RL system."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from ..config.config import DataConfig


class DataPreprocessor:
    """Handles data preprocessing for the RL system."""

    def __init__(self, config: DataConfig):
        """Initialize the preprocessor.

        Args:
            config: Data configuration
        """
        self.config = config
        self.data = None
        self.train_data = None
        self.test_data = None
        self.stats = {}

        # Sequence storage
        self.train_sequences = None
        self.train_next = None
        self.test_sequences = None
        self.test_next = None

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the data with validation."""
        try:
            # Load raw data
            df = pd.read_csv(self.config.data_path)

            # Validate required columns
            required_columns = [
                "Report Date",
                "Product Price",
                "Organic Conversion Percentage",
                "Ad Conversion Percentage",
                "Total Profit",
                "Total Sales",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert date column with validation
            try:
                df["Report Date"] = pd.to_datetime(df["Report Date"])
                df = df.sort_values("Report Date")
            except Exception as e:
                raise ValueError(f"Error processing date column: {e}")

            # Check for and handle invalid values before filling
            for col in ["Product Price", "Total Sales", "Total Profit"]:
                if (df[col] < 0).any():
                    print(f"Warning: Negative values found in {col}. Setting to 0.")
                    df.loc[df[col] < 0, col] = 0

            # Handle missing values with logging
            missing_counts = df[required_columns].isna().sum()
            if missing_counts.any():
                print("Missing value counts before filling:")
                print(missing_counts[missing_counts > 0])

            # Fill missing values with appropriate strategies
            df["Product Price"] = (
                df["Product Price"].ffill().bfill()
            )  # Forward then backward fill
            df["Organic Conversion Percentage"] = df[
                "Organic Conversion Percentage"
            ].fillna(df["Organic Conversion Percentage"].median())
            df["Ad Conversion Percentage"] = df["Ad Conversion Percentage"].fillna(
                df["Ad Conversion Percentage"].median()
            )
            df["Total Profit"] = df["Total Profit"].fillna(0)
            df["Total Sales"] = df["Total Sales"].fillna(0)

            # Add Predicted Sales if not present
            if "Predicted Sales" not in df.columns:
                df["Predicted Sales"] = (
                    df["Total Sales"].rolling(window=7, min_periods=1).mean()
                )
            # Validate conversion percentages
            for col in ["Organic Conversion Percentage", "Ad Conversion Percentage"]:
                if (df[col] > 100).any() or (df[col] < 0).any():
                    print(f"Warning: Invalid {col} values found. Clipping to [0, 100].")
                    df[col] = df[col].clip(0, 100)

            # Calculate moving averages with validation
            df["Price_MA7"] = (
                df["Product Price"].rolling(window=7, min_periods=1).mean()
            )
            df["Sales_MA7"] = df["Total Sales"].rolling(window=7, min_periods=1).mean()

            # Add time features
            df["DayOfWeek"] = df["Report Date"].dt.dayofweek
            df["Month"] = df["Report Date"].dt.month

            # Final validation
            if df.isna().any().any():
                remaining_nulls = df.isna().sum()
                print("Warning: Remaining null values after preprocessing:")
                print(remaining_nulls[remaining_nulls > 0])
                # Fill any remaining NaN values with appropriate defaults
                df = df.ffill().bfill()

            self.data = df
            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def compute_statistics(self):
        """Compute important statistics from the data."""
        self.stats = {
            "median_price": self.data["Product Price"].median(),
            "mean_price": float(self.data["Product Price"].mean()),
            "std_price": float(self.data["Product Price"].std()),
            "min_price": float(self.data["Product Price"].min()),
            "max_price": float(self.data["Product Price"].max()),
            "median_sales": self.data["Total Sales"].median(),
            "mean_sales": self.data["Total Sales"].mean(),
            "std_sales": self.data["Total Sales"].std(),
            "min_sales": self.data["Total Sales"].min(),
            "max_sales": self.data["Total Sales"].max(),
            "median_profit": self.data["Total Profit"].median(),
            "mean_profit": self.data["Total Profit"].mean(),
            "std_profit": self.data["Total Profit"].std(),
            "min_profit": self.data["Total Profit"].min(),
            "max_profit": self.data["Total Profit"].max(),
            "median_organic_conv": self.data["Organic Conversion Percentage"].median(),
            "mean_organic_conv": self.data["Organic Conversion Percentage"].mean(),
            "std_organic_conv": self.data["Organic Conversion Percentage"].std(),
            "median_ad_conv": self.data["Ad Conversion Percentage"].median(),
            "mean_ad_conv": self.data["Ad Conversion Percentage"].mean(),
            "std_ad_conv": self.data["Ad Conversion Percentage"].std(),
        }

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0,1] range with improved handling of edge cases.

        Args:
            df: DataFrame to normalize

        Returns:
            Normalized DataFrame
        """
        normalized = df.copy()

        # Normalize numerical features
        for feature in self.config.features:
            if feature in df.columns and feature != "Report Date":
                # Handle NaN values first
                if df[feature].isna().any():
                    print(
                        f"Warning: NaN values found in feature {feature}. Filling with median."
                    )
                    df[feature] = df[feature].fillna(df[feature].median())

                min_val = df[feature].min()
                max_val = df[feature].max()

                # Handle edge cases
                if np.isclose(max_val, min_val, rtol=1e-5):
                    print(
                        f"Warning: Feature {feature} has constant value {min_val}. Setting to 0.5 for better exploration."
                    )
                    normalized[feature] = 0.5
                else:
                    # Add small epsilon to avoid division by zero
                    epsilon = 1e-8
                    range_val = max_val - min_val + epsilon
                    normalized[feature] = (df[feature] - min_val) / range_val

                # Clip values to ensure they're in [0,1]
                normalized[feature] = normalized[feature].clip(0, 1)

                # Validate normalization
                if normalized[feature].isna().any():
                    print(
                        f"Warning: Normalization produced NaN values in {feature}. Setting to 0.5."
                    )
                    normalized[feature] = normalized[feature].fillna(0.5)

        return normalized

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training with validation.

        Args:
            df: DataFrame to create sequences from

        Returns:
            Tuple of (states, next_states)
        """
        sequences = []
        next_states = []

        # Validate features exist
        missing_features = [f for f in self.config.features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            return np.array([]), np.array([])

        # Ensure we have enough data for sequences
        if len(df) <= self.config.history_window:
            print(
                f"Warning: Not enough data for sequence creation. Need at least {self.config.history_window + 1} samples."
            )
            return np.array([]), np.array([])

        # Convert features to float type
        feature_df = df[self.config.features].astype(float)

        for i in range(len(df) - self.config.history_window):
            try:
                # Extract sequence and next state
                seq = feature_df.iloc[i : i + self.config.history_window].values
                next_state = feature_df.iloc[i + self.config.history_window].values

                # Validate sequence using pandas isna() for safer null checking
                if (
                    feature_df.iloc[i : i + self.config.history_window]
                    .isna()
                    .any()
                    .any()
                    or feature_df.iloc[i + self.config.history_window].isna().any()
                ):
                    print(f"Warning: Invalid sequence at index {i}. Skipping.")
                    continue

                # Check for invalid values
                if (
                    np.any(seq > 1.0)
                    or np.any(seq < 0.0)
                    or np.any(next_state > 1.0)
                    or np.any(next_state < 0.0)
                ):
                    print(
                        f"Warning: Sequence at index {i} contains values outside [0,1]. Clipping."
                    )
                    seq = np.clip(seq, 0, 1)
                    next_state = np.clip(next_state, 0, 1)

                sequences.append(seq)
                next_states.append(next_state)
            except Exception as e:
                print(f"Warning: Error processing sequence at index {i}: {e}")
                continue

        if not sequences:
            print("Warning: No valid sequences created. Check data quality.")
            return np.array([]), np.array([])

        # Convert to numpy arrays
        sequences_array = np.array(sequences)
        next_states_array = np.array(next_states)

        # Final validation
        if np.isnan(sequences_array).any() or np.isnan(next_states_array).any():
            print(
                "Warning: NaN values detected in final sequences. This should not happen."
            )
            sequences_array = np.nan_to_num(sequences_array, 0.5)
            next_states_array = np.nan_to_num(next_states_array, 0.5)

        return sequences_array, next_states_array

    def train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        split_idx = int(len(self.data) * self.config.train_test_split)
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]
        return self.train_data, self.test_data

    def get_price_history(self) -> np.ndarray:
        """Get the raw price history data.

        Returns:
            Array of historical prices
        """
        if self.data is None:
            self.load_data()

        return self.data["Product Price"].values

    def prepare_data(self) -> Dict[str, np.ndarray]:
        """Prepare data for training.

        Returns:
            Dictionary containing processed data arrays
        """
        # Load and preprocess
        self.load_data()
        self.compute_statistics()

        # Split data
        train_data, test_data = self.train_test_split()

        # Normalize
        train_normalized = self.normalize_features(train_data)
        test_normalized = self.normalize_features(test_data)

        # Create sequences
        self.train_sequences, self.train_next = self.create_sequences(train_normalized)
        self.test_sequences, self.test_next = self.create_sequences(test_normalized)

        return {
            "train_sequences": self.train_sequences,
            "train_next": self.train_next,
            "test_sequences": self.test_sequences,
            "test_next": self.test_next,
            "stats": self.stats,
        }
