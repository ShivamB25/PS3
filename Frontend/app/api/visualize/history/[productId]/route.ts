import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest, { params }: { params: { productId: string } }) {
  try {
    const response = await fetch(`${process.env.API_URL}/history/${params.productId}`)

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json(error, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Visualization history error:', error)
    return NextResponse.json(
      { detail: "Failed to fetch visualization history" },
      { status: 500 }
    )
  }
}