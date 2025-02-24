"use client"

import * as React from "react"
import { type HTMLAttributes } from "react"
import { Card } from "@/components/ui/card"
import { Badge, type BadgeProps } from "@/components/ui/badge"
import { AlertCircle, CheckCircle2, Network, Users } from "lucide-react"
import { cn } from "@/lib/utils"

interface NetworkStatusProps extends HTMLAttributes<HTMLDivElement> { }

interface NetworkState {
    connected: boolean
    mode: "private" | "consortium" | "public"
    peerCount: number
    lastSync: number
    healthStatus: string
}

export function NetworkStatus({ className, ...props }: NetworkStatusProps) {
    const [status, setStatus] = React.useState<NetworkState | null>(null)
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        const fetchStatus = async () => {
            try {
                const response = await fetch("/api/network/status")
                if (!response.ok) {
                    throw new Error("Failed to fetch network status")
                }
                const data = await response.json()
                setStatus(data)
                setError(null)
            } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to fetch status")
            }
        }

        // Initial fetch
        fetchStatus()

        // Poll every 5 seconds
        const interval = setInterval(fetchStatus, 5000)
        return () => clearInterval(interval)
    }, [])

    if (error) {
        return (
            <Card className={cn("p-4", className)} {...props}>
                <div className="flex items-center space-x-2 text-destructive">
                    <AlertCircle className="h-4 w-4" />
                    <span>Network Error: {error}</span>
                </div>
            </Card>
        )
    }

    if (!status) {
        return (
            <Card className={cn("p-4", className)} {...props}>
                <div className="flex items-center space-x-2">
                    <Network className="h-4 w-4 animate-pulse" />
                    <span>Loading network status...</span>
                </div>
            </Card>
        )
    }

    return (
        <Card className={cn("p-4", className)} {...props}>
            <div className="space-y-4">
                <div className="flex items-center justify-between">
                    <h3 className="font-semibold">Network Status</h3>
                    <Badge variant={status.connected ? "default" : "destructive"}>
                        {status.connected ? "Connected" : "Disconnected"}
                    </Badge>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center space-x-2">
                        <Network className="h-4 w-4" />
                        <span>Mode: {status.mode}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                        <Users className="h-4 w-4" />
                        <span>Peers: {status.peerCount}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                        <CheckCircle2 className="h-4 w-4" />
                        <span>Health: {status.healthStatus}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                        <span>Last Sync: {new Date(status.lastSync).toLocaleTimeString()}</span>
                    </div>
                </div>
            </div>
        </Card>
    )
}
