"""FastAPI implementation for price optimization with CSV handling and model registry."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

from ..config.config import Config
from ..data.preprocessor import DataPreprocessor
from ..models.sac_agent import SACAgent
from ..env.price_env import PriceOptimizationEnv
from ..train import train

router = APIRouter()


class TrainingConfig(BaseModel):
    """Training configuration."""

    exploration_rate: float = 1.0
    price_increase_bias: float = 0.2
    num_episodes: int = 1000


class PredictionConfig(BaseModel):
    """Prediction configuration."""

    future_dates: List[str]
    exploration_mode: bool = False
    exploration_rate: float = 0.1


class ModelRegistry:
    """Manages trained models and their metadata."""

    def __init__(self, base_dir: str = "model_registry"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

    def get_product_dir(self, product_id: str) -> Path:
        """Get product directory."""
        product_dir = self.base_dir / product_id
        product_dir.mkdir(exist_ok=True, parents=True)
        return product_dir

    def ensure_product_dir(self, product_id: str) -> Path:
        """Ensure product directory exists and return path."""
        return self.get_product_dir(product_id)

    def save_training_data(self, product_id: str, df: pd.DataFrame):
        """Save training data for future reference."""
        product_dir = self.ensure_product_dir(product_id)
        df.to_csv(product_dir / "training_data.csv", index=False)

    def save_predictions(self, product_id: str, predictions: List[Dict]):
        """Save predictions including historical and future data."""
        product_dir = self.ensure_product_dir(product_id)
        with open(product_dir / "predictedhistory.json", "w") as f:
            json.dump(predictions, f, indent=2)

    def get_training_data(self, product_id: str) -> Optional[pd.DataFrame]:
        """Get training data for a product."""
        try:
            product_dir = self.get_product_dir(product_id)
            training_data_path = product_dir / "training_data.csv"
            if training_data_path.exists():
                df = pd.read_csv(training_data_path)
                df["Report Date"] = pd.to_datetime(df["Report Date"])
                return df
        except Exception as e:
            print(f"Error reading training data: {e}")
        return None

    def get_latest_model(self, product_id: str) -> Optional[str]:
        """Get path to best model checkpoint."""
        product_dir = self.get_product_dir(product_id)
        best_model = product_dir / "best_model.pt"
        if best_model.exists():
            return str(best_model)
        return None

    def get_metadata(self, product_id: str) -> Optional[Dict]:
        """Get model metadata."""
        try:
            product_dir = self.get_product_dir(product_id)
            with open(product_dir / "metadata.json", "r") as f:
                return json.load(f)
        except:
            return None

    def list_products(self) -> List[Dict]:
        """List all products with their metadata."""
        products = []
        for product_dir in self.base_dir.iterdir():
            if product_dir.is_dir():
                metadata = self.get_metadata(product_dir.name)
                if metadata:
                    # Extract model name from the product_id, replacing underscores/hyphens with spaces
                    # and capitalizing each word for better readability
                    model_name = " ".join(
                        word.capitalize()
                        for word in product_dir.name.replace("_", " ").replace("-", " ").split()
                    )
                    products.append({
                        "product_id": product_dir.name,
                        "name": model_name,  # Add human-readable name
                        "metadata": metadata
                    })
        return products


class PriceOptimizer:
    """Handles price optimization with CSV data."""

    def __init__(self):
        self.config = Config()
        self.registry = ModelRegistry()

    async def predict(self, product_id: str, config: PredictionConfig) -> List[Dict]:
        """Generate predictions using training data."""
        try:
            # Verify model exists
            model_path = self.registry.get_latest_model(product_id)
            if not model_path:
                raise HTTPException(
                    status_code=404, detail=f"No trained model found for {product_id}"
                )

            # Get training data
            df = self.registry.get_training_data(product_id)
            if df is None:
                raise HTTPException(
                    status_code=404, detail=f"No training data found for {product_id}"
                )

            # Save temporary data
            temp_csv = f"temp_pred_{product_id}.csv"
            df.to_csv(temp_csv, index=False)

            # Initialize components
            self.config.data.data_path = temp_csv
            preprocessor = DataPreprocessor(self.config.data)
            preprocessor.prepare_data()

            env = PriceOptimizationEnv(preprocessor, self.config.env)

            # Initialize agent
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            price_history = preprocessor.get_price_history()

            agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                config=self.config.model,
                price_history=price_history,
            )

            # Load model
            agent.load(model_path)

            # Set exploration parameters
            agent.price_memory.exploration_rate = (
                config.exploration_rate if config.exploration_mode else 0.1
            )

            # Generate predictions
            predictions = []
            state, _ = env.reset()

            for date in config.future_dates:
                # Get action
                action = agent.select_action(
                    state, evaluate=not config.exploration_mode
                )

                # Get temporal context
                state_reshaped = state.reshape(-1, len(preprocessor.config.features))
                current_day = int(state_reshaped[-1][8] * 7)
                current_month = int(state_reshaped[-1][9] * 12) + 1

                # Apply temporal factors
                day_factor = 1.0 + 0.05 * np.cos(2 * np.pi * current_day / 7)
                month_factor = 1.0 + 0.025 * np.sin(2 * np.pi * current_month / 12)

                # Calculate price
                base_price = env.min_price + ((action[0] + 1) / 2) * env.price_range
                adjusted_price = base_price * day_factor * month_factor

                # Get normalized action
                adjusted_action = action.copy()
                adjusted_action[0] = (
                    2 * (adjusted_price - env.min_price) / env.price_range
                ) - 1
                adjusted_action = np.clip(adjusted_action, -1.0, 1.0)

                # Get prediction
                next_state, reward, done, _, info = env.step(adjusted_action)

                # Get exploration info
                exploration_bonus = agent.price_memory.get_exploration_bonus(
                    adjusted_price
                )
                exploration_std = agent.price_memory.get_optimal_exploration_std(
                    adjusted_price
                )

                predictions.append(
                    {
                        "date": date,
                        "recommended_price": float(adjusted_price),
                        "predicted_sales": float(info["sales"]),
                        "metrics": {
                            "organic_conversion": float(info.get("organic_conv", 0)),
                            "ad_conversion": float(info.get("ad_conv", 0)),
                            "predicted_profit": float(info.get("profit", 0)),
                        },
                        "exploration_info": {
                            "exploration_bonus": float(exploration_bonus),
                            "exploration_std": float(exploration_std),
                            "price_vs_median": float(
                                adjusted_price / preprocessor.stats["median_price"]
                            ),
                            "is_new_price_point": float(exploration_bonus > 0.5),
                        },
                    }
                )

                if done:
                    state, _ = env.reset()
                else:
                    state = next_state

            # Cleanup
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

            return predictions

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Global optimizer instance
optimizer = PriceOptimizer()


@router.post("/predict/{product_id}")
async def predict_prices(product_id: str, config: PredictionConfig):
    """Generate predictions using training data."""
    try:
        predictions = await optimizer.predict(product_id, config)
        return JSONResponse(content=predictions)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models")
async def list_models():
    """List all trained models."""
    return optimizer.registry.list_products()


@router.get("/models/{product_id}")
async def get_model_info(product_id: str):
    """Get model information."""
    metadata = optimizer.registry.get_metadata(product_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model not found for {product_id}")
    return metadata


@router.get("/history/{product_id}")
async def get_prediction_history(product_id: str):
    """Get prediction history for a product."""
    try:
        # First check if product directory exists
        product_dir = optimizer.registry.get_product_dir(product_id)
        if not product_dir.exists():
            raise HTTPException(
                status_code=404, detail=f"No data found for product {product_id}"
            )

        # Try to get prediction history
        history_path = product_dir / "predictedhistory.json"
        if not history_path.exists():
            # If no predictions exist yet, return empty list
            return []

        with open(history_path, "r") as f:
            history = json.load(f)

        return history
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get prediction history: {str(e)}"
        )


@router.post("/train/{product_id}")
async def train_model(product_id: str, file: UploadFile = File(...)):
    """Train model and generate predictions."""
    temp_csv = f"temp_train_{product_id}.csv"
    try:
        # Save uploaded file
        with open(temp_csv, "wb") as f:
            content = await file.read()
            f.write(content)

        # Read and validate CSV
        df = pd.read_csv(temp_csv)
        df["Report Date"] = pd.to_datetime(df["Report Date"])

        # Save training data
        optimizer.registry.save_training_data(product_id, df)

        # Configure training
        optimizer.config.data.data_path = temp_csv
        optimizer.config.training.checkpoint_dir = str(
            optimizer.registry.get_product_dir(product_id)
        )

        # Train model
        metrics = await train(optimizer.config)

        # Create and save metadata
        product_dir = optimizer.registry.ensure_product_dir(product_id)
        metadata = {
            "last_trained": datetime.now().isoformat(),
            "data_stats": {
                "total_records": len(df),
                "price_records": int(df["Product Price"].notna().sum()),
                "sales_records": int(df["Total Sales"].notna().sum()),
            },
            "performance": {
                "mae": metrics.get("mae", 0.0),
                "rmse": metrics.get("rmse", 0.0),
                "r2_score": metrics.get("r2_score", 0.0),
            },
            "config": {
                "random_seed": optimizer.config.training.random_seed,
                "batch_size": optimizer.config.model.batch_size,
                "actor_learning_rate": optimizer.config.model.actor_learning_rate,
                "critic_learning_rate": optimizer.config.model.critic_learning_rate,
            },
        }
        with open(product_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Initialize predictor with best model
        model_path = optimizer.registry.get_latest_model(product_id)
        if not model_path:
            raise HTTPException(
                status_code=500, detail="Training failed to produce a model"
            )

        # Initialize components for prediction
        preprocessor = DataPreprocessor(optimizer.config.data)
        preprocessor.prepare_data()
        env = PriceOptimizationEnv(preprocessor, optimizer.config.env)

        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        price_history = preprocessor.get_price_history()
        agent = SACAgent(state_dim, action_dim, optimizer.config.model, price_history)
        agent.load(model_path)

        # Generate predictions for missing data with state continuity
        training_predictions = []
        state, _ = env.reset()  # Reset only once at the start
        for _, row in df.iterrows():
            if pd.isna(row["Product Price"]) or pd.isna(row["Total Sales"]):
                action = agent.select_action(state, evaluate=True)
                state, reward, done, _, info = env.step(action)
                if done:  # Reset if episode is done
                    state, _ = env.reset()

                base_price = env.min_price + ((action[0] + 1) / 2) * env.price_range
                training_predictions.append(
                    {
                        "date": row["Report Date"].strftime("%Y-%m-%d"),
                        "recommended_price": (
                            float(base_price)
                            if pd.isna(row["Product Price"])
                            else float(row["Product Price"])
                        ),
                        "predicted_sales": (
                            float(info["sales"])
                            if pd.isna(row["Total Sales"])
                            else float(row["Total Sales"])
                        ),
                        "predicted_organic_conv": float(info.get("organic_conv", 0)),
                        "predicted_ad_conv": float(info.get("ad_conv", 0)),
                        "is_historical": True,
                    }
                )

        # Generate predictions with state continuity
        current_date = datetime.now()
        start_date = current_date - timedelta(days=30)

        # Reset environment once for past predictions
        state, _ = env.reset()
        past_predictions = []
        for i in range(30):
            pred_date = start_date + timedelta(days=i)
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _, info = env.step(action)
            if done:  # Reset if episode is done
                state, _ = env.reset()

            base_price = env.min_price + ((action[0] + 1) / 2) * env.price_range
            past_predictions.append(
                {
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "recommended_price": float(base_price),
                    "predicted_sales": float(info["sales"]),
                    "predicted_organic_conv": float(info.get("organic_conv", 0)),
                    "predicted_ad_conv": float(info.get("ad_conv", 0)),
                    "is_historical": True,
                }
            )

        # Reset environment once for future predictions
        state, _ = env.reset()
        future_predictions = []
        for i in range(30):
            pred_date = current_date + timedelta(days=i)
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _, info = env.step(action)
            if done:  # Reset if episode is done
                state, _ = env.reset()

            base_price = env.min_price + ((action[0] + 1) / 2) * env.price_range
            future_predictions.append(
                {
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "recommended_price": float(base_price),
                    "predicted_sales": float(info["sales"]),
                    "predicted_organic_conv": float(info.get("organic_conv", 0)),
                    "predicted_ad_conv": float(info.get("ad_conv", 0)),
                    "is_historical": False,
                }
            )

        # Combine all predictions
        all_predictions = training_predictions + past_predictions + future_predictions

        # Sort by date and save
        all_predictions.sort(key=lambda x: x["date"])
        optimizer.registry.save_predictions(product_id, all_predictions)

        # Cleanup
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        return {
            "message": "Training and prediction completed successfully",
            "metrics": metrics,
            "predictions_count": len(all_predictions),
        }

    except Exception as e:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
