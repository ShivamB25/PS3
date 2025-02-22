import { type NextRequest, NextResponse } from "next/server"
import { type Model } from "@/types/api"

export async function GET() {
  try {
    const response = await fetch(`${process.env.API_URL}/models`, {
      headers: {
        'Accept': 'application/json',
      },
    })

    if (!response.ok) {
      console.error('Failed to fetch models:', response.status, response.statusText)
      const error = await response.json()
      return NextResponse.json(error, { status: response.status })
    }

    const backendData = await response.json()
    
    // Transform backend data to frontend format
    const transformedData: Model[] = backendData.map((item: any) => ({
      product_id: item.product_id,
      name: `Price Optimizer - ${item.product_id}`,
      metadata: {
        training_date: item.metadata.last_trained,
        data_range: {
          start: item.metadata.data_stats.date_range?.start || '',
          end: item.metadata.data_stats.date_range?.end || ''
        },
        data_stats: {
          total_records: item.metadata.data_stats.total_records,
          price_records: item.metadata.data_stats.price_records,
          sales_records: item.metadata.data_stats.sales_records,
          predicted_sales_records: item.metadata.data_stats.sales_records // Using sales_records as a proxy
        },
        training_config: {
          exploration_rate: item.metadata.training_info?.hyperparameters?.exploration_rate || 1.0,
          price_increase_bias: item.metadata.training_info?.hyperparameters?.price_increase_bias || 0.2,
          num_episodes: item.metadata.training_info?.epochs || 1000
        },
        performance: {
          ...item.metadata.performance,
          model_type: "SAC",
          architecture: item.metadata.model_info?.architecture || {}
        }
      }
    }))

    console.log('Transformed models data:', transformedData) // Debug log
    return NextResponse.json(transformedData satisfies Model[])
  } catch (error) {
    console.error('Failed to fetch models:', error)
    return NextResponse.json(
      { detail: "Failed to fetch models" },
      { status: 500 }
    )
  }
}