import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest, { params }: { params: { productId: string } }) {
  try {
    const formData = await request.formData()
    
    const response = await fetch(`${process.env.API_URL}/visualize/predictions/${params.productId}`, {
      method: "POST",
      body: formData
    })

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json(error, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Prediction visualization error:', error)
    return NextResponse.json(
      { detail: "Failed to generate prediction visualization" },
      { status: 500 }
    )
  }
}