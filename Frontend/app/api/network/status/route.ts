import { NextResponse } from "next/server"

export async function GET() {
    try {
        const response = await fetch("http://localhost:8000/network/status")
        if (!response.ok) {
            throw new Error("Failed to fetch network status")
        }
        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error("Network status error:", error)
        return NextResponse.json(
            { error: "Failed to fetch network status" },
            { status: 500 }
        )
    }
}
