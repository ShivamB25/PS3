import type { ModelMetadata, PriceData, PredictionResponse } from "@/types/api"

export const DEMO_MODELS: Record<string, ModelMetadata> = {
  woolball: {
    training_date: "2025-02-20T10:00:00Z",
    data_range: {
      start: "2024-10-01",
      end: "2025-02-20",
    },
    data_stats: {
      total_records: 143,
      price_records: 143,
      sales_records: 143,
      predicted_sales_records: 140,
    },
    training_config: {
      exploration_rate: 1.0,
      price_increase_bias: 0.2,
      num_episodes: 1000,
    },
    performance: {
      mae: 12.5,
      rmse: 15.3,
      r2_score: 0.85,
    },
  },
  "cotton-shirt": {
    training_date: "2025-02-19T15:30:00Z",
    data_range: {
      start: "2024-09-01",
      end: "2025-02-19",
    },
    data_stats: {
      total_records: 172,
      price_records: 172,
      sales_records: 172,
      predicted_sales_records: 165,
    },
    training_config: {
      exploration_rate: 0.8,
      price_increase_bias: 0.15,
      num_episodes: 1200,
    },
    performance: {
      mae: 8.2,
      rmse: 10.1,
      r2_score: 0.89,
    },
  },
}

export const DEMO_HISTORICAL_DATA: Record<string, PriceData[]> = {
  woolball: [
    { date: "2024-10-01", price: 19.99, sales: 120 },
    { date: "2024-11-01", price: 21.99, sales: 105 },
    { date: "2024-12-01", price: 24.99, sales: 150 },
    { date: "2025-01-01", price: 22.99, sales: 135 },
    { date: "2025-02-01", price: 20.99, sales: 142 },
  ],
  "cotton-shirt": [
    { date: "2024-09-01", price: 29.99, sales: 80 },
    { date: "2024-10-01", price: 32.99, sales: 65 },
    { date: "2024-11-01", price: 27.99, sales: 95 },
    { date: "2024-12-01", price: 34.99, sales: 120 },
    { date: "2025-01-01", price: 29.99, sales: 88 },
  ],
}

export function generateDemoPredictions(productId: string, dates: string[]): PredictionResponse[] {
  const basePrice = DEMO_HISTORICAL_DATA[productId]?.slice(-1)[0]?.price || 20
  const baseSales = DEMO_HISTORICAL_DATA[productId]?.slice(-1)[0]?.sales || 100

  return dates.map((date) => ({
    date,
    recommended_price: basePrice * (0.9 + Math.random() * 0.2),
    predicted_sales: baseSales * (0.8 + Math.random() * 0.4),
    metrics: {
      organic_conversion: 2.5,
      ad_conversion: 1.8,
      predicted_profit: 750.0,
    },
    exploration_info: {
      exploration_bonus: 0.1,
      exploration_std: 0.05,
      price_vs_median: 1.02,
      is_new_price_point: false,
    },
  }))
}

