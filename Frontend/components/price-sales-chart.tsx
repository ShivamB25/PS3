"use client"

import React from "react"
import { Card } from "@/components/ui/card"
import {
  LineChart,
  Line,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { TooltipProps } from "recharts"
import { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent"
import { format } from "date-fns"

interface HistoryDataPoint {
  date: string
  recommended_price: number
  predicted_sales: number
  predicted_organic_conv: number
  predicted_ad_conv: number
  is_historical: boolean
}

interface PriceSalesChartProps {
  data: HistoryDataPoint[]
}

interface ChartData extends HistoryDataPoint {
  formattedDate: string
}

export function PriceSalesChart({ data }: PriceSalesChartProps) {
  const chartData = React.useMemo(() => {
    return data.map(point => ({
      ...point,
      formattedDate: format(new Date(point.date), 'yyyy-MM-dd')
    }));
  }, [data])

  const CustomTooltip = ({ active, payload, label }: TooltipProps<ValueType, NameType>) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload as HistoryDataPoint
      const isHistorical = data.is_historical
      return (
        <div className="rounded-lg border bg-white/90 backdrop-blur-sm p-3 shadow-lg">
          <div className="text-sm font-medium">
            {format(new Date(data.date), 'MMM d, yyyy')}
            <span className="ml-2 text-xs text-muted-foreground">
              {isHistorical ? '(Historical)' : '(Predicted)'}
            </span>
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            Price: ${data.recommended_price.toFixed(2)}
          </div>
          <div className="text-xs text-muted-foreground">
            Sales: {Math.round(data.predicted_sales)} units
          </div>
          <div className="text-xs text-muted-foreground">
            Organic Conv: {data.predicted_organic_conv.toFixed(2)}%
          </div>
          <div className="text-xs text-muted-foreground">
            Ad Conv: {data.predicted_ad_conv.toFixed(2)}%
          </div>
        </div>
      )
    }
    return null
  }

  // Split data into historical and predicted
  const historicalData = chartData.filter(d => d.is_historical)
  const predictedData = chartData.filter(d => !d.is_historical)

  return (
    <Card className="p-6 bg-white">
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-gray-900">
          Product Performance
        </h3>
        <p className="text-sm text-gray-500 mt-1">Historical data and predictions</p>
      </div>

      {/* Price & Sales Chart */}
      <div className="mb-2">
        <div className="text-sm font-medium text-gray-700 mb-1">Price & Sales Trends</div>
      </div>
      <div className="h-[200px] mb-8">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart 
            margin={{ top: 10, right: 30, bottom: 0, left: 30 }}
          >
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="#f3f4f6" 
              horizontal={true}
              vertical={false}
            />

            <XAxis
              dataKey="formattedDate"
              tickFormatter={(date) => {
                if (!date) return '';
                try {
                  return format(new Date(date), 'MMM d');
                } catch (e) {
                  console.error('Invalid date:', date);
                  return '';
                }
              }}
              stroke="#9ca3af"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              interval="preserveStartEnd"
              domain={['dataMin', 'dataMax']}
              type="category"
              allowDataOverflow={false}
            />

            <YAxis
              yAxisId="price"
              stroke="#9ca3af"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              tickFormatter={(value: number) => `$${value.toFixed(2)}`}
              domain={['dataMin - 1', 'dataMax + 1']}
            />

            <YAxis
              yAxisId="sales"
              orientation="right"
              stroke="#9ca3af"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              tickFormatter={(value: number) => value.toString()}
            />

            <Tooltip content={CustomTooltip} />

            {/* Historical Price Line */}
            <Line
              yAxisId="price"
              type="monotone"
              data={historicalData}
              dataKey="recommended_price"
              stroke="#4f46e5"
              strokeWidth={2}
              dot={{ fill: "#4f46e5", r: 2, strokeWidth: 0 }}
              activeDot={{ r: 4, strokeWidth: 0 }}
              name="Historical Price"
              isAnimationActive={true}
              animationDuration={1000}
            />

            {/* Historical Sales Line */}
            <Line
              yAxisId="sales"
              type="monotone"
              data={historicalData}
              dataKey="predicted_sales"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ fill: "#10b981", r: 2, strokeWidth: 0 }}
              activeDot={{ r: 4, strokeWidth: 0 }}
              name="Historical Sales"
              isAnimationActive={true}
              animationDuration={1000}
            />

            {/* Predicted Price Line */}
            <Line
              yAxisId="price"
              type="monotone"
              data={predictedData}
              dataKey="recommended_price"
              stroke="#dc2626"
              strokeWidth={3}
              dot={{ fill: "#dc2626", r: 4, strokeWidth: 0 }}
              activeDot={{ r: 6, strokeWidth: 0 }}
              name="Predicted Price"
              isAnimationActive={true}
              animationDuration={1000}
            />

            {/* Predicted Sales Line */}
            <Line
              yAxisId="sales"
              type="monotone"
              data={predictedData}
              dataKey="predicted_sales"
              stroke="#ea580c"
              strokeWidth={3}
              dot={{ fill: "#ea580c", r: 4, strokeWidth: 0 }}
              activeDot={{ r: 6, strokeWidth: 0 }}
              name="Predicted Sales"
              isAnimationActive={true}
              animationDuration={1000}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Conversion Rates Chart */}
      <div className="mb-2">
        <div className="text-sm font-medium text-gray-700 mb-1">Conversion Rates</div>
      </div>
      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart 
            data={chartData}
            margin={{ top: 10, right: 30, bottom: 0, left: 30 }}
          >
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="#f3f4f6"
              horizontal={true}
              vertical={false}
            />

            <XAxis
              dataKey="formattedDate"
              tickFormatter={(date) => {
                if (!date) return '';
                try {
                  return format(new Date(date), 'MMM d');
                } catch (e) {
                  console.error('Invalid date:', date);
                  return '';
                }
              }}
              stroke="#9ca3af"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              interval="preserveStartEnd"
              domain={['dataMin', 'dataMax']}
              type="category"
              allowDataOverflow={false}
            />

            <YAxis
              stroke="#9ca3af"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              tickFormatter={(value: number) => `${value.toFixed(1)}%`}
            />

            <Tooltip content={CustomTooltip} />

            <Line
              type="monotone"
              dataKey="predicted_organic_conv"
              stroke="#0891b2"
              strokeWidth={2}
              dot={{ fill: "#0891b2", r: 2, strokeWidth: 0 }}
              activeDot={{ r: 4, strokeWidth: 0 }}
              name="Organic Conversion"
            />

            <Line
              type="monotone"
              dataKey="predicted_ad_conv"
              stroke="#6366f1"
              strokeWidth={2}
              dot={{ fill: "#6366f1", r: 2, strokeWidth: 0 }}
              activeDot={{ r: 4, strokeWidth: 0 }}
              name="Ad Conversion"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-6 text-sm text-gray-600 border-t pt-4">
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-emerald-500" />
          <span>Historical Sales</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-indigo-600" />
          <span>Historical Price</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-[#ea580c]" />
          <span>Predicted Sales</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-[#dc2626]" />
          <span>Predicted Price</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-[#0891b2]" />
          <span>Organic Conversion</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-[#6366f1]" />
          <span>Ad Conversion</span>
        </div>
      </div>
    </Card>
  )
}
