import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest, { params }: { params: { productId: string } }) {
  console.log(`[Visualization API] Fetching model visualization for product: ${params.productId}`)
  
  try {
    const apiUrl = `${process.env.API_URL}/viz/model/${params.productId}`
    console.log(`[Visualization API] Making request to: ${apiUrl}`)
    
    const response = await fetch(apiUrl)
    console.log(`[Visualization API] Response status: ${response.status}`)

    if (!response.ok) {
      const error = await response.json()
      console.error('[Visualization API] Model visualization error:', {
        status: response.status,
        error,
        productId: params.productId,
        endpoint: apiUrl
      })
      return NextResponse.json({
        detail: error.detail || "Failed to fetch model visualization",
        status: response.status,
        productId: params.productId
      }, { status: response.status })
    }

    const data = await response.json()
    console.log(`[Visualization API] Successfully fetched visualization data for product: ${params.productId}`)
    return NextResponse.json(data)
  } catch (error) {
    console.error('[Visualization API] Unexpected error:', {
      error,
      productId: params.productId,
      message: error instanceof Error ? error.message : 'Unknown error'
    })
    return NextResponse.json(
      { detail: "Failed to fetch model visualization" },
      { status: 500 }
    )
  }
}