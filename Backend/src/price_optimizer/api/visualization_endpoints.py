"""Visualization endpoints for price optimization API."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from typing import List
import json
from pathlib import Path
import logging

from .visualization import PriceVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class ModelRegistryError(Exception):
    """Custom exception for model registry errors."""

    pass


def get_model_registry_path(product_id: str) -> Path:
    """Get path to model registry for a product."""
    registry_path = Path("model_registry") / product_id
    if not registry_path.exists():
        raise ModelRegistryError(f"No model registry found for product {product_id}")
    return registry_path


@router.post("/historical/{product_id}")
async def visualize_historical(product_id: str, file: UploadFile = None):
    """Generate historical analysis visualizations.

    Creates:
    1. Time series of price and sales
    2. Price vs sales scatter plot with trend line
    3. Conversion rate analysis (if data available)
    """
    logger.info(f"Received historical visualization request for product: {product_id}")

    try:
        # Verify product exists in registry
        logger.info(f"Checking model registry for product: {product_id}")
        registry_path = get_model_registry_path(product_id)
        logger.info(f"Found model registry at: {registry_path}")

        # Read data - either from uploaded file or training data
        if file:
            logger.info("Reading uploaded CSV data")
            df = pd.read_csv(file.file)
        else:
            logger.info("No file provided, using training data")
            training_path = registry_path / "training_data.csv"
            if not training_path.exists():
                raise ModelRegistryError(
                    f"No training data found for product {product_id}"
                )
            df = pd.read_csv(training_path)

        df["Report Date"] = pd.to_datetime(df["Report Date"])
        logger.info(f"Successfully loaded data with {len(df)} records")

        # Generate visualizations
        logger.info("Initializing PriceVisualizer")
        visualizer = PriceVisualizer()

        # Save visualization data for future reference
        viz_path = registry_path / "visualizations"
        viz_path.mkdir(exist_ok=True)
        logger.info(f"Created/verified visualizations directory at: {viz_path}")

        logger.info("Saving historical data CSV")
        df.to_csv(viz_path / "historical_data.csv", index=False)

        logger.info("Generating visualization data")
        visualizations = {
            "historical_analysis": visualizer.create_historical_analysis(df),
            "price_sales_scatter": visualizer.create_price_sales_scatter(df),
            "conversion_analysis": visualizer.create_conversion_analysis(df),
        }
        logger.info("Successfully generated visualization data")

        # Save visualization data
        logger.info("Saving visualization data to JSON")
        with open(viz_path / "historical_analysis.json", "w") as f:
            json.dump(visualizations, f)
        logger.info("Successfully saved visualization data")

        return visualizations

    except ModelRegistryError as e:
        logger.error(f"Model registry error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during visualization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.post("/predictions/{product_id}")
async def visualize_predictions(
    predictions: List[dict], product_id: str, file: UploadFile = File(...)
):
    """Generate prediction analysis visualizations.

    Creates:
    1. Historical vs predicted price and sales
    2. Exploration analysis showing price recommendations and exploration bonuses
    """
    logger.info(f"Received predictions visualization request for product: {product_id}")

    try:
        # Verify product exists in registry
        logger.info(f"Checking model registry for product: {product_id}")
        registry_path = get_model_registry_path(product_id)
        logger.info(f"Found model registry at: {registry_path}")

        # Read historical data
        logger.info("Reading historical data CSV")
        df = pd.read_csv(file.file)
        df["Report Date"] = pd.to_datetime(df["Report Date"])
        logger.info(f"Successfully loaded historical data with {len(df)} records")

        # Generate visualizations
        logger.info("Initializing PriceVisualizer")
        visualizer = PriceVisualizer()

        # Save visualization data
        viz_path = registry_path / "visualizations"
        viz_path.mkdir(exist_ok=True)
        logger.info(f"Created/verified visualizations directory at: {viz_path}")

        # Save prediction data
        logger.info("Saving prediction data to CSV")
        pred_df = pd.DataFrame(predictions)
        prediction_file = (
            f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        pred_df.to_csv(viz_path / prediction_file, index=False)
        logger.info(f"Saved predictions to {prediction_file}")

        logger.info("Generating visualization data")
        visualizations = {
            "prediction_analysis": visualizer.create_prediction_visualization(
                df, predictions
            ),
            "exploration_analysis": visualizer.create_exploration_analysis(predictions),
        }
        logger.info("Successfully generated visualization data")

        # Save visualization data
        logger.info("Saving visualization data to JSON")
        with open(viz_path / "prediction_analysis.json", "w") as f:
            json.dump(visualizations, f)
        logger.info("Successfully saved visualization data")

        return visualizations

    except ModelRegistryError as e:
        logger.error(f"Model registry error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during visualization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.get("/model/{product_id}")
async def visualize_model_performance(product_id: str):
    """Generate model performance visualizations.

    Creates:
    1. Training rewards over time
    2. Price exploration coverage
    3. Sales prediction accuracy
    """
    logger.info(f"Received model visualization request for product: {product_id}")

    try:
        # Verify product exists and load data
        logger.info(f"Checking model registry for product: {product_id}")
        registry_path = get_model_registry_path(product_id)
        logger.info(f"Found model registry at: {registry_path}")

        # Load model metadata
        try:
            logger.info(f"Loading metadata for product: {product_id}")
            with open(registry_path / "metadata.json", "r") as f:
                metadata = json.load(f)
            logger.info("Successfully loaded metadata")
        except FileNotFoundError:
            logger.error(f"Metadata not found for product: {product_id}")
            raise ModelRegistryError(f"No metadata found for product {product_id}")

        # Load training data
        try:
            logger.info(f"Loading training data for product: {product_id}")
            df = pd.read_csv(registry_path / "training_data.csv")
            df["Report Date"] = pd.to_datetime(df["Report Date"])
            logger.info(f"Successfully loaded training data with {len(df)} records")
        except FileNotFoundError:
            logger.error(f"Training data not found for product: {product_id}")
            raise ModelRegistryError(f"No training data found for product {product_id}")

        # Generate visualizations
        logger.info("Initializing PriceVisualizer")
        visualizer = PriceVisualizer()

        # Save visualization data
        viz_path = registry_path / "visualizations"
        viz_path.mkdir(exist_ok=True)
        logger.info(f"Created/verified visualizations directory at: {viz_path}")

        logger.info("Generating visualization data")
        visualizations = {
            "historical_analysis": visualizer.create_historical_analysis(df),
            "training_performance": metadata.get("performance", {}),
            "data_coverage": {
                "total_records": int(len(df)),
                "price_records": int(df["Product Price"].notna().sum()),
                "sales_records": int(df["Total Sales"].notna().sum()),
                "conversion_records": {
                    "organic": int(
                        df["Organic Conversion Percentage"].notna().sum()
                        if "Organic Conversion Percentage" in df
                        else 0
                    ),
                    "ad": int(
                        df["Ad Conversion Percentage"].notna().sum()
                        if "Ad Conversion Percentage" in df
                        else 0
                    ),
                },
            },
        }
        logger.info("Successfully generated visualization data")

        # Save visualization data
        logger.info("Saving visualization data to JSON")
        with open(viz_path / "model_performance.json", "w") as f:
            json.dump(visualizations, f)
        logger.info("Successfully saved visualization data")

        return visualizations

    except ModelRegistryError as e:
        logger.error(f"Model registry error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during visualization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.get("/history/{product_id}")
async def get_visualization_history(product_id: str):
    """Get all saved visualizations for a product."""
    logger.info(f"Received visualization history request for product: {product_id}")

    try:
        # Verify product exists
        logger.info(f"Checking model registry for product: {product_id}")
        registry_path = get_model_registry_path(product_id)
        viz_path = registry_path / "visualizations"
        logger.info(f"Looking for visualizations at: {viz_path}")

        if not viz_path.exists():
            logger.info(f"No visualization directory found for product: {product_id}")
            return {"visualizations": []}

        # Get all visualization files
        logger.info("Loading visualization history")
        history = {
            "historical_analysis": (
                json.loads(viz_path.joinpath("historical_analysis.json").read_text())
                if viz_path.joinpath("historical_analysis.json").exists()
                else None
            ),
            "prediction_analysis": (
                json.loads(viz_path.joinpath("prediction_analysis.json").read_text())
                if viz_path.joinpath("prediction_analysis.json").exists()
                else None
            ),
            "model_performance": (
                json.loads(viz_path.joinpath("model_performance.json").read_text())
                if viz_path.joinpath("model_performance.json").exists()
                else None
            ),
            "prediction_history": [f.stem for f in viz_path.glob("predictions_*.csv")],
        }

        logger.info("Successfully loaded visualization history")
        logger.info(
            f"Found files: historical_analysis: {'Yes' if history['historical_analysis'] else 'No'}, "
            + f"prediction_analysis: {'Yes' if history['prediction_analysis'] else 'No'}, "
            + f"model_performance: {'Yes' if history['model_performance'] else 'No'}, "
            + f"prediction_files: {len(history['prediction_history'])}"
        )

        return history

    except ModelRegistryError as e:
        logger.error(f"Model registry error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"Unexpected error while getting visualization history: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get visualization history: {str(e)}"
        )
