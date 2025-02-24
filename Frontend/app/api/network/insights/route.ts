import { NextResponse } from "next/server"

export async function GET() {
    try {
        const response = await fetch("http://localhost:8000/network/insights")
        if (!response.ok) {
            throw new Error("Failed to fetch network insights")
        }
        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error("Network insights error:", error)
        return NextResponse.json(
            {
                error: "Failed to fetch network insights",
                insights: [],
                peerCount: 0,
                marketHealth: 0,
                recentActivity: []
            },
            { status: 500 }
        )
    }
}

// Optional: Add endpoint for specific product insights
export async function POST(request: Request) {
    try {
        const { productId } = await request.json()
        const response = await fetch(`http://localhost:8000/network/insights/${productId}`)
        if (!response.ok) {
            throw new Error("Failed to fetch product insights")
        }
        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error("Product insights error:", error)
        return NextResponse.json(
            {
                error: "Failed to fetch product insights",
                insights: [],
                peerCount: 0,
                marketHealth: 0,
                recentActivity: []
            },
            { status: 500 }
        )
    }
}
