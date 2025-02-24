"use client"

import * as React from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"

interface NetworkConfigProps extends React.HTMLAttributes<HTMLDivElement> { }

interface NetworkSettings {
    mode: "private" | "consortium" | "public"
    privacy: {
        anonymizeData: boolean
        encryptConnection: boolean
        dataSharing: {
            priceData: "full" | "ranges" | "none"
            salesData: "full" | "aggregated" | "none"
            trends: "full" | "aggregated" | "none"
        }
    }
}

export function NetworkConfig({ className, ...props }: NetworkConfigProps) {
    const [settings, setSettings] = React.useState<NetworkSettings>({
        mode: "private",
        privacy: {
            anonymizeData: true,
            encryptConnection: true,
            dataSharing: {
                priceData: "ranges",
                salesData: "aggregated",
                trends: "full"
            }
        }
    })

    const handleModeChange = async (mode: NetworkSettings["mode"]) => {
        try {
            const response = await fetch("/api/network/config", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    mode,
                    privacy: settings.privacy
                })
            })

            if (!response.ok) {
                throw new Error("Failed to update network mode")
            }

            setSettings(prev => ({ ...prev, mode }))
        } catch (err) {
            console.error("Failed to update network mode:", err)
        }
    }

    const handlePrivacyChange = async (key: keyof NetworkSettings["privacy"], value: boolean) => {
        try {
            const response = await fetch("/api/network/config", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    mode: settings.mode,
                    privacy: {
                        ...settings.privacy,
                        [key]: value
                    }
                })
            })

            if (!response.ok) {
                throw new Error("Failed to update privacy settings")
            }

            setSettings(prev => ({
                ...prev,
                privacy: {
                    ...prev.privacy,
                    [key]: value
                }
            }))
        } catch (err) {
            console.error("Failed to update privacy settings:", err)
        }
    }

    const handleDataSharingChange = async (
        key: keyof NetworkSettings["privacy"]["dataSharing"],
        value: "full" | "ranges" | "aggregated" | "none"
    ) => {
        try {
            const response = await fetch("/api/network/config", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    mode: settings.mode,
                    privacy: {
                        ...settings.privacy,
                        dataSharing: {
                            ...settings.privacy.dataSharing,
                            [key]: value
                        }
                    }
                })
            })

            if (!response.ok) {
                throw new Error("Failed to update data sharing settings")
            }

            setSettings(prev => ({
                ...prev,
                privacy: {
                    ...prev.privacy,
                    dataSharing: {
                        ...prev.privacy.dataSharing,
                        [key]: value
                    }
                }
            }))
        } catch (err) {
            console.error("Failed to update data sharing settings:", err)
        }
    }

    return (
        <Card className={cn("p-6", className)} {...props}>
            <div className="space-y-6">
                <div className="space-y-2">
                    <h3 className="text-lg font-semibold">Network Configuration</h3>
                    <p className="text-sm text-muted-foreground">
                        Configure your P2P network settings and privacy preferences.
                    </p>
                </div>

                <div className="space-y-4">
                    <div className="space-y-2">
                        <Label>Network Mode</Label>
                        <Select value={settings.mode} onValueChange={handleModeChange}>
                            <SelectTrigger>
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="private">Private (Single Organization)</SelectItem>
                                <SelectItem value="consortium">Consortium (Trusted Group)</SelectItem>
                                <SelectItem value="public">Public (Open Market)</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    <div className="space-y-4">
                        <Label>Privacy Settings</Label>
                        <div className="space-y-2">
                            <div className="flex items-center justify-between">
                                <Label htmlFor="anonymize-data" className="cursor-pointer">
                                    Anonymize Data
                                </Label>
                                <Switch
                                    id="anonymize-data"
                                    checked={settings.privacy.anonymizeData}
                                    onCheckedChange={(checked) => handlePrivacyChange("anonymizeData", checked)}
                                />
                            </div>
                            <div className="flex items-center justify-between">
                                <Label htmlFor="encrypt-connection" className="cursor-pointer">
                                    Encrypt Connection
                                </Label>
                                <Switch
                                    id="encrypt-connection"
                                    checked={settings.privacy.encryptConnection}
                                    onCheckedChange={(checked) => handlePrivacyChange("encryptConnection", checked)}
                                />
                            </div>
                        </div>
                    </div>

                    <div className="space-y-4">
                        <Label>Data Sharing</Label>
                        <div className="space-y-2">
                            <div className="grid gap-2">
                                <Label>Price Data</Label>
                                <Select
                                    value={settings.privacy.dataSharing.priceData}
                                    onValueChange={(value: any) => handleDataSharingChange("priceData", value)}
                                >
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="full">Full Data</SelectItem>
                                        <SelectItem value="ranges">Price Ranges Only</SelectItem>
                                        <SelectItem value="none">No Sharing</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <div className="grid gap-2">
                                <Label>Sales Data</Label>
                                <Select
                                    value={settings.privacy.dataSharing.salesData}
                                    onValueChange={(value: any) => handleDataSharingChange("salesData", value)}
                                >
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="full">Full Data</SelectItem>
                                        <SelectItem value="aggregated">Aggregated Only</SelectItem>
                                        <SelectItem value="none">No Sharing</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <div className="grid gap-2">
                                <Label>Market Trends</Label>
                                <Select
                                    value={settings.privacy.dataSharing.trends}
                                    onValueChange={(value: any) => handleDataSharingChange("trends", value)}
                                >
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="full">Full Data</SelectItem>
                                        <SelectItem value="aggregated">Aggregated Only</SelectItem>
                                        <SelectItem value="none">No Sharing</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </Card>
    )
}
