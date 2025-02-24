"use client"

import { useEffect, useState } from "react"
import { PriceSalesChart } from "@/components/price-sales-chart"
import { PredictionForm } from "@/components/prediction-form"
import { ModelHistory } from "@/components/model-history"
import { NetworkStatus } from "@/components/network-status"
import { NetworkConfig } from "@/components/network-config"
import { NetworkInsights } from "@/components/network-insights"
import { api } from "@/lib/api"
import type { ModelMetadata, PredictionRequest } from "@/types/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"
import { Card } from "@/components/ui/card"
import { format } from "date-fns"

interface HistoryDataPoint {
  date: string
  recommended_price: number
  predicted_sales: number
  predicted_organic_conv: number
  predicted_ad_conv: number
  is_historical: boolean
}

export default function Home() {
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null)
  const [modelData, setModelData] = useState<Record<string, ModelMetadata>>({})
  const [historyData, setHistoryData] = useState<HistoryDataPoint[]>([])
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [debug, setDebug] = useState<string>("")

  // Load available models
  useEffect(() => {
    async function loadModels() {
      try {
        setIsLoading(true)
        setError(null)
        const models = await api.listModels()
        if (Array.isArray(models)) {
          const modelMap: Record<string, ModelMetadata> = {}
          for (const model of models) {
            if (model.product_id && model.metadata) {
              modelMap[model.product_id] = model.metadata
            }
          }
          setModelData(modelMap)
        } else {
          console.error('Invalid models response:', models)
          setError('Invalid models data received')
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load models")
      } finally {
        setIsLoading(false)
      }
    }

    loadModels()
  }, [])

  // Load history data when product changes
  useEffect(() => {
    async function loadProductData() {
      if (!selectedProductId || !modelData[selectedProductId]) {
        setHistoryData([])
        return
      }

      try {
        setIsLoading(true)
        setError(null)

        // Get history data
        const response = await fetch(`/api/visualize/history/${selectedProductId}`)
        if (!response.ok) {
          throw new Error('Failed to fetch history data')
        }

        const data = await response.json()
        console.log('Received history data:', data)
        setHistoryData(data)
      } catch (err) {
        console.error('Error loading history data:', err)
        setError(err instanceof Error ? err.message : "Failed to load data")
        setHistoryData([])
      } finally {
        setIsLoading(false)
      }
    }

    loadProductData()
  }, [selectedProductId, modelData])

  async function handlePrediction(request: PredictionRequest) {
    if (!selectedProductId) {
      setError("No product selected")
      return
    }

    try {
      setIsLoading(true)
      setError(null)
      setDebug("")

      // Log the prediction request config
      const debugInfo = [
        "=== Request Config ===",
        JSON.stringify({
          future_dates: request.future_dates.map(d => format(new Date(d), 'yyyy-MM-dd')),
          exploration_mode: request.exploration_mode,
          exploration_rate: request.exploration_rate
        }, null, 2)
      ].join('\n')

      setDebug(debugInfo)


      // Make prediction request
      const predictionResponse = await fetch(`/api/predict/${selectedProductId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request)
      })

      if (!predictionResponse.ok) {
        const error = await predictionResponse.text()
        throw new Error(`Failed to generate predictions: ${error}`)
      }

      // Get updated history data including predictions
      const historyResponse = await fetch(`/api/visualize/history/${selectedProductId}`)
      if (!historyResponse.ok) {
        throw new Error('Failed to fetch updated history data')
      }

      const data = await historyResponse.json()
      console.log('Received updated history data:', data)
      setHistoryData(data)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err instanceof Error ? err.message : "Failed to generate predictions")
    } finally {
      setIsLoading(false)
    }
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          {error}
          {debug && (
            <pre className="mt-4 p-4 bg-gray-900 text-gray-100 rounded-lg overflow-auto text-xs">
              {debug}
            </pre>
          )}
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <main className="container mx-auto space-y-6 p-4">
      <ModelHistory
        models={modelData}
        onSelectModel={setSelectedProductId}
        selectedProductId={selectedProductId}
        onTrainingComplete={async () => {
          try {
            setIsLoading(true)
            setError(null)
            const models = await api.listModels()
            const modelMap: Record<string, ModelMetadata> = {}
            for (const model of models) {
              modelMap[model.product_id] = model.metadata
            }
            setModelData(modelMap)
          } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load models")
          } finally {
            setIsLoading(false)
          }
        }}
      />

      {selectedProductId && modelData[selectedProductId] ? (
        <>
          <div className="grid gap-6 md:grid-cols-[2fr_1fr]">
            <PriceSalesChart data={historyData} />

            <Card className="p-6">
              <h3 className="mb-4 text-lg font-semibold">Generate Predictions</h3>
              <PredictionForm onSubmit={handlePrediction} isLoading={isLoading} />
              {debug && (
                <pre className="mt-4 p-4 bg-gray-100 rounded-lg overflow-auto text-xs">
                  {debug}
                </pre>
              )}
            </Card>
          </div>

          <div className="mt-8 space-y-6">
            <h2 className="text-2xl font-bold">P2P Network</h2>
            <div className="grid gap-6 md:grid-cols-[1fr_2fr]">
              <div className="space-y-6">
                <NetworkStatus />
                <NetworkConfig />
              </div>
              <NetworkInsights />
            </div>
          </div>
        </>
      ) : (
        <Card className="p-6">
          <h3 className="text-lg font-semibold">No Model Selected</h3>
          <p className="text-sm text-muted-foreground mt-2">
            Please select a trained model from the list above to view its performance and generate predictions.
          </p>
        </Card>
      )}
    </main>
  )
}
