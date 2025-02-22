"""Main entry point for the price optimization system."""

import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from price_optimizer.config.config import Config
from price_optimizer.train import train


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Price Optimization RL System")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="Mode to run the system in",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint for inference"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/woolballhistory.csv",
        help="Path to data file (e.g., data/woolballhistory.csv or data/soapnutshistory.csv)",
    )
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_args()

    # Create config
    config = Config()
    config.training.random_seed = args.seed
    config.data.data_path = args.data_path

    if args.mode == "train":
        await train(config)
        # After training, run inference with the best model
        from price_optimizer.inference import PricePredictor
        import json

        # Use the best model for predictions
        checkpoint_path = os.path.join(config.training.checkpoint_dir, "best_model.pt")

        # Initialize predictor with best model
        predictor = PricePredictor(config, checkpoint_path)

        # Read and process training data
        import pandas as pd

        df = pd.read_csv(config.data.data_path)
        df["Report Date"] = pd.to_datetime(df["Report Date"])

        # Generate predictions for missing data in training period
        training_predictions = []
        for _, row in df.iterrows():
            if pd.isna(row["Product Price"]) or pd.isna(row["Total Sales"]):
                state, _ = predictor.env.reset()
                pred = predictor.predict_price(state)
                training_predictions.append(
                    {
                        "date": row["Report Date"].strftime("%Y-%m-%d"),
                        "recommended_price": (
                            pred["recommended_price"]
                            if pd.isna(row["Product Price"])
                            else float(row["Product Price"])
                        ),
                        "predicted_sales": (
                            pred["predicted_sales"]
                            if pd.isna(row["Total Sales"])
                            else float(row["Total Sales"])
                        ),
                        "predicted_organic_conv": pred["predicted_organic_conv"],
                        "predicted_ad_conv": pred["predicted_ad_conv"],
                        "is_historical": True,
                    }
                )

        # Generate predictions for one month before current date
        current_date = datetime.now()
        start_date = current_date - timedelta(days=30)
        past_predictions = []
        for i in range(30):
            pred_date = start_date + timedelta(days=i)
            state, _ = predictor.env.reset()
            pred = predictor.predict_price(state)
            past_predictions.append(
                {
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "recommended_price": pred["recommended_price"],
                    "predicted_sales": pred["predicted_sales"],
                    "predicted_organic_conv": pred["predicted_organic_conv"],
                    "predicted_ad_conv": pred["predicted_ad_conv"],
                    "is_historical": True,
                }
            )

        # Generate predictions for one month after current date
        future_predictions = []
        for i in range(30):
            pred_date = current_date + timedelta(days=i)
            state, _ = predictor.env.reset()
            pred = predictor.predict_price(state)
            future_predictions.append(
                {
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "recommended_price": pred["recommended_price"],
                    "predicted_sales": pred["predicted_sales"],
                    "predicted_organic_conv": pred["predicted_organic_conv"],
                    "predicted_ad_conv": pred["predicted_ad_conv"],
                    "is_historical": False,
                }
            )

        # Combine all predictions
        all_predictions = training_predictions + past_predictions + future_predictions

        # Sort by date
        all_predictions.sort(key=lambda x: x["date"])

        # Save predictions to JSON
        with open("predictedhistory.json", "w") as f:
            json.dump(all_predictions, f, indent=2)

        print("\nPredictions saved to predictedhistory.json")
    else:
        from price_optimizer.inference import run_inference

        if not args.checkpoint:
            raise ValueError("Checkpoint path must be provided for inference mode")
        run_inference(config, args.checkpoint)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
