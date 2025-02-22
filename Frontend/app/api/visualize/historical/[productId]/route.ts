import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest, { params }: { params: { productId: string } }) {
  console.log(`[Visualization API] Processing historical visualization for product: ${params.productId}`)
  
  try {
    const apiUrl = `${process.env.API_URL}/historical/${params.productId}`
    let requestOptions: RequestInit = { method: "POST" }

    // Only include form data if a file was provided
    try {
      const formData = await request.formData()
      const file = formData.get('file')
      if (file) {
        console.log('[Visualization API] Using provided CSV file')
        requestOptions.body = formData
      } else {
        console.log('[Visualization API] No file provided, using training data')
      }
    } catch (error) {
      console.log('[Visualization API] No form data, using training data')
    }

    console.log(`[Visualization API] Making request to: ${apiUrl}`)
    const response = await fetch(apiUrl, requestOptions)
    console.log(`[Visualization API] Response status: ${response.status}`)

    if (!response.ok) {
      const error = await response.json()
      console.error('[Visualization API] Historical visualization error:', {
        status: response.status,
        error,
        productId: params.productId,
        endpoint: apiUrl
      })
      return NextResponse.json({
        detail: error.detail || "Failed to generate historical visualization",
        status: response.status,
        productId: params.productId
      }, { status: response.status })
    }

    const data = await response.json()
    console.log(`[Visualization API] Successfully generated visualization data for product: ${params.productId}`)
    return NextResponse.json(data)
  } catch (error) {
    console.error('[Visualization API] Unexpected error:', {
      error,
      productId: params.productId,
      message: error instanceof Error ? error.message : 'Unknown error'
    })
    return NextResponse.json(
      { detail: "Failed to generate historical visualization" },
      { status: 500 }
    )
  }
}