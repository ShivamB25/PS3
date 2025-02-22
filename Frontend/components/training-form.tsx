import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { useToast } from "@/components/ui/use-toast"

interface TrainingFormProps {
  onTrainingComplete: () => void
}

export function TrainingForm({ onTrainingComplete }: TrainingFormProps) {
  const [productId, setProductId] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!productId || !file) {
      toast({
        title: "Missing fields",
        description: "Please provide both product ID and CSV file",
        variant: "destructive",
      })
      return
    }

    setIsLoading(true)
    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch(`/api/train/${productId}`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || "Failed to train model")
      }

      const result = await response.json()
      const metrics = result.metadata?.performance || {}
      
      const metricsText = Object.entries(metrics)
        .filter(([key]) => key !== 'model_type' && key !== 'architecture')
        .map(([key, value]) => `${key}: ${typeof value === 'number' ? value.toFixed(4) : value}`)
        .join(', ')

      toast({
        title: "Training completed",
        description: `Model trained successfully. ${metricsText ? `Metrics: ${metricsText}` : ''}`,
      })

      onTrainingComplete()
    } catch (error) {
      toast({
        title: "Training failed",
        description: error instanceof Error ? error.message : "Failed to train model",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
      setProductId("")
      setFile(null)
    }
  }

  return (
    <Card className="p-6">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">Train New Model</h3>
        <p className="text-sm text-muted-foreground">Upload historical data to train a new pricing model</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <Input
            placeholder="Product ID"
            value={productId}
            onChange={(e) => setProductId(e.target.value)}
            className="mb-2"
          />
          <Input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="mb-2"
          />
        </div>
        <Button type="submit" disabled={isLoading}>
          {isLoading ? "Training..." : "Train Model"}
        </Button>
      </form>
    </Card>
  )
}