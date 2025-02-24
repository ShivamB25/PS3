"use client"

import * as React from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown, LineChart, Users, Activity } from "lucide-react"
import { cn } from "@/lib/utils"

interface NetworkInsightsProps extends React.HTMLAttributes<HTMLDivElement> { }

interface MarketInsight {
    priceRange: {
        min: number
        max: number
    }
    trend: number
    confidence: number
    timestamp: number
    sourceType: "private" | "consortium" | "public"
}

interface NetworkInsightData {
    insights: MarketInsight[]
    peerCount: number
    marketHealth: number
    recentActivity: {
        timestamp: number
        type: string
        value: number
    }[]
}

export function NetworkInsights({ className, ...props }: NetworkInsightsProps) {
    const [data, setData] = React.useState<NetworkInsightData | null>(null)
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        const fetchInsights = async () => {
            try {
                const response = await fetch("/api/network/insights")
                if (!response.ok) {
                    throw new Error("Failed to fetch network insights")
                }
                const data = await response.json()
                setData(data)
                setError(null)
            } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to fetch insights")
            }
        }

        // Initial fetch
        fetchInsights()

        // Poll every 30 seconds
        const interval = setInterval(fetchInsights, 30000)
        return () => clearInterval(interval)
    }, [])

    if (error) {
        return (
            <Card className={cn("p-4", className)} {...props}>
                <div className="text-sm text-destructive">
                    Error loading insights: {error}
                </div>
            </Card>
        )
    }

    if (!data) {
        return (
            <Card className={cn("p-4", className)} {...props}>
                <div className="flex items-center justify-center h-40">
                    <div className="animate-pulse text-muted-foreground">
                        Loading insights...
                    </div>
                </div>
            </Card>
        )
    }

    const averagePrice = data.insights.reduce((sum, insight) => {
        return sum + (insight.priceRange.min + insight.priceRange.max) / 2
    }, 0) / data.insights.length

    const averageTrend = data.insights.reduce((sum, insight) => {
        return sum + insight.trend
    }, 0) / data.insights.length

    return (
        <Card className={cn("p-6", className)} {...props}>
            <div className="space-y-6">
                <div className="space-y-2">
                    <h3 className="text-lg font-semibold">Network Insights</h3>
                    <p className="text-sm text-muted-foreground">
                        Market intelligence from connected peers
                    </p>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">Average Price</span>
                            <Badge variant="outline">
                                ${averagePrice.toFixed(2)}
                            </Badge>
                        </div>
                        <div className="flex items-center space-x-2">
                            {averageTrend > 0 ? (
                                <TrendingUp className="h-4 w-4 text-green-500" />
                            ) : (
                                <TrendingDown className="h-4 w-4 text-red-500" />
                            )}
                            <span className="text-sm text-muted-foreground">
                                {Math.abs(averageTrend * 100).toFixed(1)}% {averageTrend > 0 ? "up" : "down"}
                            </span>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">Network Activity</span>
                            <Badge variant="outline">
                                <Users className="h-4 w-4 mr-1" />
                                {data.peerCount} peers
                            </Badge>
                        </div>
                        <div className="flex items-center space-x-2">
                            <Activity className="h-4 w-4" />
                            <span className="text-sm text-muted-foreground">
                                {data.recentActivity.length} updates in last hour
                            </span>
                        </div>
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h4 className="text-sm font-medium">Recent Market Signals</h4>
                        <LineChart className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <div className="space-y-2">
                        {data.recentActivity.slice(0, 5).map((activity, i) => (
                            <div
                                key={i}
                                className="flex items-center justify-between text-sm"
                            >
                                <span className="text-muted-foreground">
                                    {new Date(activity.timestamp).toLocaleTimeString()}
                                </span>
                                <span>{activity.type}</span>
                                <Badge variant="outline">
                                    ${activity.value.toFixed(2)}
                                </Badge>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="pt-4 border-t">
                    <div className="text-sm text-muted-foreground">
                        Market health score: {(data.marketHealth * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        </Card>
    )
}
