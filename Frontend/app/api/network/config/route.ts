import { NextRequest, NextResponse } from "next/server"

export async function GET() {
    try {
        const response = await fetch("http://localhost:8000/network/config")
        if (!response.ok) {
            throw new Error("Failed to fetch network configuration")
        }
        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error("Network config error:", error)
        return NextResponse.json(
            { error: "Failed to fetch network configuration" },
            { status: 500 }
        )
    }
}

export async function POST(request: NextRequest) {
    try {
        const config = await request.json()

        const response = await fetch("http://localhost:8000/network/config", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(config),
        })

        if (!response.ok) {
            throw new Error("Failed to update network configuration")
        }

        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error("Network config update error:", error)
        return NextResponse.json(
            { error: "Failed to update network configuration" },
            { status: 500 }
        )
    }
}
