export interface Model {
  product_id: string
  name: string
  metadata: ModelMetadata
}

export interface ModelMetadata {
  training_date: string
  data_range: {
    start: string
    end: string
  }
  data_stats: {
    total_records: number
    price_records: number
    sales_records: number
    predicted_sales_records: number
  }
  training_config: {
    exploration_rate: number
    price_increase_bias: number
    num_episodes: number
  }
  performance: Record<string, any>
}

export interface HistoryDataPoint {
  date: string
  recommended_price: number
  predicted_sales: number
  predicted_organic_conv: number
  predicted_ad_conv: number
  is_historical: boolean
}

export interface PredictionRequest {
  future_dates: string[]
  exploration_mode: boolean
  exploration_rate: number
}

export interface PredictionResponse {
  date: string
  recommended_price: number
  predicted_sales: number
  metrics: {
    organic_conversion: number
    ad_conversion: number
    predicted_profit: number
  }
  exploration_info: {
    exploration_bonus: number
    exploration_std: number
    price_vs_median: number
    is_new_price_point: boolean
  }
}

export interface PlotlyFigure {
  data: any[]
  layout: Record<string, any>
}

export interface HistoricalVisualization {
  historical_analysis: PlotlyFigure
  price_sales_scatter: PlotlyFigure
  conversion_analysis: PlotlyFigure
}

export interface PredictionVisualization {
  prediction_analysis: PlotlyFigure
  exploration_analysis: PlotlyFigure
}

export interface ModelPerformanceVisualization {
  historical_analysis: PlotlyFigure
  training_performance: Record<string, any>
  data_coverage: {
    total_records: number
    price_records: number
    sales_records: number
    conversion_records: {
      organic: number
      ad: number
    }
  }
}

export interface VisualizationHistory {
  historical_analysis: PlotlyFigure
  prediction_analysis: PlotlyFigure
  model_performance: Record<string, any>
  prediction_history: string[]
}

export interface ApiError {
  detail: string
  model_path?: string
  metadata?: ModelMetadata
}

export interface TrainingResponse {
  status: string
  product_id: string
  metadata: ModelMetadata
}
