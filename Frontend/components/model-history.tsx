import { Card } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { ModelMetadata } from "@/types/api"
import { Badge } from "@/components/ui/badge"
import { formatDistanceToNow } from "date-fns"
import { TrainingForm } from "./training-form"

interface ModelHistoryProps {
  models: Record<string, ModelMetadata>
  onSelectModel: (productId: string) => void
  selectedProductId: string | null
  onTrainingComplete: () => void
}

export function ModelHistory({ models, onSelectModel, selectedProductId, onTrainingComplete }: ModelHistoryProps) {
  return (
    <>
      <TrainingForm onTrainingComplete={onTrainingComplete} />
      
      <Card className="p-6">
        <div className="mb-4">
          <h3 className="text-lg font-semibold">Trained Models</h3>
          <p className="text-sm text-muted-foreground">Select a model to view its performance and generate predictions</p>
        </div>

        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Product ID</TableHead>
                <TableHead>Training Date</TableHead>
                <TableHead>Records</TableHead>
                <TableHead>Performance</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(models).map(([productId, model]) => (
                <TableRow
                  key={productId}
                  className={
                    productId === selectedProductId ? "bg-muted/50 cursor-pointer" : "cursor-pointer hover:bg-muted/50"
                  }
                  onClick={() => onSelectModel(productId)}
                >
                  <TableCell className="font-medium">{productId}</TableCell>
                  <TableCell>
                    {(() => {
                      try {
                        const date = new Date(model.training_date);
                        if (isNaN(date.getTime())) {
                          return "Invalid date";
                        }
                        return formatDistanceToNow(date, {
                          addSuffix: true,
                        });
                      } catch (error) {
                        return "Invalid date";
                      }
                    })()}
                  </TableCell>
                  <TableCell>{model.data_stats?.total_records || 'N/A'}</TableCell>
                  <TableCell>
                    {(() => {
                      const metrics = model.performance || {};
                      if ('r2_score' in metrics) {
                        return `R² = ${metrics.r2_score.toFixed(2)}`;
                      } else if ('mae' in metrics) {
                        return `MAE = ${metrics.mae.toFixed(4)}`;
                      }
                      return 'N/A';
                    })()}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="bg-primary/10">
                      Active
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>
    </>
  )
}
