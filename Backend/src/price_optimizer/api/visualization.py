"""Visualization utilities for price optimization analysis."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import json


class PriceVisualizer:
    """Generates visualizations for price optimization analysis."""

    @staticmethod
    def create_historical_analysis(df: pd.DataFrame) -> Dict:
        """Create historical price vs sales analysis."""
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df["Report Date"],
                y=df["Product Price"],
                name="Price",
                line=dict(color="blue"),
                mode="lines+markers",
            ),
            secondary_y=False,
        )

        # Add sales line
        fig.add_trace(
            go.Scatter(
                x=df["Report Date"],
                y=df["Total Sales"],
                name="Sales",
                line=dict(color="green"),
                mode="lines+markers",
            ),
            secondary_y=True,
        )

        # Add predicted sales if available
        if "Predicted Sales" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Report Date"],
                    y=df["Predicted Sales"],
                    name="Predicted Sales",
                    line=dict(color="red", dash="dot"),
                    mode="lines",
                ),
                secondary_y=True,
            )

        # Update layout
        fig.update_layout(
            title="Historical Price and Sales Analysis",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2_title="Sales",
            hovermode="x unified",
        )

        return json.loads(fig.to_json())

    @staticmethod
    def create_price_sales_scatter(df: pd.DataFrame) -> Dict:
        """Create price vs sales scatter plot with trend analysis."""
        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=df["Product Price"],
                y=df["Total Sales"],
                mode="markers",
                name="Data Points",
                marker=dict(
                    size=8,
                    color=df["Total Profit"] if "Total Profit" in df.columns else None,
                    colorscale="Viridis",
                    showscale=True if "Total Profit" in df.columns else False,
                    colorbar=(
                        dict(title="Profit") if "Total Profit" in df.columns else None
                    ),
                ),
            )
        )

        # Add trend line
        mask = ~df["Product Price"].isna() & ~df["Total Sales"].isna()
        if mask.any():
            x = df.loc[mask, "Product Price"]
            y = df.loc[mask, "Total Sales"]
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            x_range = np.linspace(x.min(), x.max(), 100)

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    name="Trend",
                    line=dict(color="red", dash="dash"),
                )
            )

        fig.update_layout(
            title="Price vs Sales Relationship",
            xaxis_title="Price",
            yaxis_title="Sales",
            hovermode="closest",
        )

        return json.loads(fig.to_json())

    @staticmethod
    def create_conversion_analysis(df: pd.DataFrame) -> Dict:
        """Create conversion rate analysis."""
        fig = make_subplots(rows=2, cols=1)

        # Add organic conversion
        if "Organic Conversion Percentage" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Report Date"],
                    y=df["Organic Conversion Percentage"],
                    name="Organic Conversion",
                    line=dict(color="blue"),
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )

        # Add ad conversion
        if "Ad Conversion Percentage" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Report Date"],
                    y=df["Ad Conversion Percentage"],
                    name="Ad Conversion",
                    line=dict(color="orange"),
                    mode="lines+markers",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(height=800, title="Conversion Rate Analysis", showlegend=True)

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Organic Conversion %", row=1, col=1)
        fig.update_yaxes(title_text="Ad Conversion %", row=2, col=1)

        return json.loads(fig.to_json())

    @staticmethod
    def create_prediction_visualization(
        historical_df: pd.DataFrame, predictions: List[Dict]
    ) -> Dict:
        """Create visualization of predictions vs historical data."""
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)
        pred_df["date"] = pd.to_datetime(pred_df["date"])

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Historical data
        fig.add_trace(
            go.Scatter(
                x=historical_df["Report Date"],
                y=historical_df["Product Price"],
                name="Historical Price",
                line=dict(color="blue"),
                mode="lines+markers",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=historical_df["Report Date"],
                y=historical_df["Total Sales"],
                name="Historical Sales",
                line=dict(color="green"),
                mode="lines+markers",
            ),
            secondary_y=True,
        )

        # Predictions
        fig.add_trace(
            go.Scatter(
                x=pred_df["date"],
                y=pred_df["recommended_price"],
                name="Predicted Price",
                line=dict(color="red", dash="dot"),
                mode="lines+markers",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=pred_df["date"],
                y=pred_df["predicted_sales"],
                name="Predicted Sales",
                line=dict(color="orange", dash="dot"),
                mode="lines+markers",
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            title="Price and Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2_title="Sales",
            hovermode="x unified",
        )

        return json.loads(fig.to_json())

    @staticmethod
    def create_exploration_analysis(predictions: List[Dict]) -> Dict:
        """Create visualization of exploration vs exploitation."""
        pred_df = pd.DataFrame(predictions)
        pred_df["date"] = pd.to_datetime(pred_df["date"])

        fig = make_subplots(rows=2, cols=1)

        # Price and exploration bonus
        fig.add_trace(
            go.Scatter(
                x=pred_df["date"],
                y=pred_df["recommended_price"],
                name="Recommended Price",
                line=dict(color="blue"),
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=pred_df["date"],
                y=[d["exploration_info"]["exploration_bonus"] for d in predictions],
                name="Exploration Bonus",
                line=dict(color="red"),
                mode="lines+markers",
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(height=800, title="Exploration Analysis", showlegend=True)

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Exploration Bonus", row=2, col=1)

        return json.loads(fig.to_json())
