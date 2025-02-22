import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest, { params }: { params: { productId: string } }) {
  try {
    const formData = await request.formData()
    const force = request.nextUrl.searchParams.get('force') === 'true'
    
    const url = new URL(`/train/${params.productId}`, process.env.API_URL)
    if (force) {
      url.searchParams.set('force', 'true')
    }
    
    const response = await fetch(url, {
      method: "POST",
      body: formData
    })

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json(error, { status: response.status })
    }

    const backendData = await response.json()
    
    // Transform backend response to frontend format
    // Handle both detailed and simple response formats
    const transformedData = {
      status: "success",
      product_id: params.productId,
      metadata: backendData.metadata ? {
        // If detailed metadata is available
        training_date: backendData.metadata.last_trained,
        data_range: {
          start: backendData.metadata.data_stats?.date_range?.start || '',
          end: backendData.metadata.data_stats?.date_range?.end || ''
        },
        data_stats: {
          total_records: backendData.metadata.data_stats?.total_records || backendData.predictions_count || 0,
          price_records: backendData.metadata.data_stats?.price_records || backendData.predictions_count || 0,
          sales_records: backendData.metadata.data_stats?.sales_records || backendData.predictions_count || 0,
          predicted_sales_records: backendData.metadata.data_stats?.sales_records || backendData.predictions_count || 0
        },
        training_config: {
          exploration_rate: backendData.metadata.training_info?.hyperparameters?.exploration_rate || 1.0,
          price_increase_bias: backendData.metadata.training_info?.hyperparameters?.price_increase_bias || 0.2,
          num_episodes: backendData.metadata.training_info?.epochs || 1000
        },
        performance: {
          ...(backendData.metadata?.performance || backendData.metrics || {}),
          model_type: "SAC",
          architecture: backendData.metadata?.model_info?.architecture || {}
        }
      } : {
        // Simple format
        training_date: new Date().toISOString(),
        data_range: { start: '', end: '' },
        data_stats: {
          total_records: backendData.predictions_count || 0,
          price_records: backendData.predictions_count || 0,
          sales_records: backendData.predictions_count || 0,
          predicted_sales_records: backendData.predictions_count || 0
        },
        training_config: {
          exploration_rate: 1.0,
          price_increase_bias: 0.2,
          num_episodes: 1000
        },
        performance: {
          ...backendData.metrics,
          model_type: "SAC"
        }
      }
    }

    return NextResponse.json(transformedData)
  } catch (error) {
    console.error('Training error:', error)
    return NextResponse.json(
      { detail: "Failed to train model" },
      { status: 500 }
    )
  }
}