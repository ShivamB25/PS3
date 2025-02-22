import {
  ApiError,
  ModelMetadata,
  PredictionRequest,
  PredictionResponse,
  TrainingResponse,
  HistoryDataPoint,
  ModelPerformanceVisualization
} from '../types/api'

// Use the public URL for client-side requests
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const emitApiLog = (log: {
  type: 'request' | 'response' | 'error';
  method: string;
  endpoint: string;
  data?: any;
  duration?: string;
  status?: number;
}) => {
  if (typeof window !== 'undefined') {
    const event = new CustomEvent('api-log', {
      detail: {
        ...log,
        timestamp: Date.now()
      }
    });
    window.dispatchEvent(event);
  }
};

class ApiClient {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const startTime = performance.now()
    const method = options.method || 'GET'
    
    emitApiLog({
      type: 'request',
      method,
      endpoint,
      data: options.body ? JSON.parse(options.body as string) : undefined
    })

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          ...options.headers,
        }
      })

      const endTime = performance.now()
      const duration = `${(endTime - startTime).toFixed(2)}ms`
      const data = await response.json()
      
      if (!response.ok) {
        emitApiLog({
          type: 'error',
          method,
          endpoint,
          data,
          duration,
          status: response.status
        })
        throw data as ApiError
      }

      emitApiLog({
        type: 'response',
        method,
        endpoint,
        data,
        duration,
        status: response.status
      })

      return data as T
    } catch (error) {
      const endTime = performance.now()
      const duration = `${(endTime - startTime).toFixed(2)}ms`

      if (error instanceof TypeError) {
        emitApiLog({
          type: 'error',
          method,
          endpoint,
          data: { message: error.message },
          duration
        })
      }
      throw error
    }
  }

  // Model Management
  async listModels(): Promise<Array<{ product_id: string; metadata: ModelMetadata }>> {
    try {
      return await this.request<Array<{ product_id: string; metadata: ModelMetadata }>>('/models')
    } catch (error) {
      throw error
    }
  }

  async checkModel(productId: string): Promise<ModelMetadata> {
    try {
      return await this.request<ModelMetadata>(`/models/${productId}`)
    } catch (error) {
      throw error
    }
  }

  async getHistoricalData(productId: string): Promise<Array<{
    date: string;
    recommended_price: number;
    predicted_sales: number;
    predicted_organic_conv: number;
    predicted_ad_conv: number;
    is_historical: boolean;
  }>> {
    try {
      return await this.request<any[]>(`/history/${productId}`)
    } catch (error) {
      throw error
    }
  }

  async getModelPerformance(productId: string): Promise<ModelPerformanceVisualization> {
    try {
      return await this.request<ModelPerformanceVisualization>(`/model/${productId}`)
    } catch (error) {
      throw error
    }
  }

  async getPredictions(
    productId: string, 
    config: {
      future_dates: string[];
      exploration_mode?: boolean;
      exploration_rate?: number;
    }
  ): Promise<PredictionResponse[]> {
    try {
      return await this.request<PredictionResponse[]>(`/predict/${productId}`, {
        method: 'POST',
        body: JSON.stringify(config)
      })
    } catch (error) {
      throw error
    }
  }

  // Training endpoints
  async trainModel(productId: string, config: any): Promise<TrainingResponse> {
    try {
      return await this.request<TrainingResponse>(`/train/${productId}`, {
        method: 'POST',
        body: JSON.stringify(config)
      })
    } catch (error) {
      throw error
    }
  }
}

// Export singleton instance
export const api = new ApiClient()

// Error handling utilities
export function isApiError(error: unknown): error is ApiError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'detail' in error &&
    typeof (error as ApiError).detail === 'string'
  )
}

export function getErrorMessage(error: unknown): string {
  if (isApiError(error)) {
    return error.detail
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unknown error occurred'
}
