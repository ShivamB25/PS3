"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { format } from "date-fns"

import { Button } from "@/components/ui/button"
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import type { PredictionRequest } from "@/types/api"

const formSchema = z.object({
  dates: z.string().min(1, "Please select at least one date"),
  exploration_mode: z.boolean().default(false),
  exploration_rate: z.number().min(0).max(1).default(0.1),
})

interface PredictionFormProps {
  onSubmit: (data: PredictionRequest) => void
  isLoading: boolean
}

export function PredictionForm({ onSubmit, isLoading }: PredictionFormProps) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      dates: format(new Date(), "yyyy-MM-dd"),
      exploration_mode: false,
      exploration_rate: 0.1,
    },
  })

  function handleSubmit(values: z.infer<typeof formSchema>) {
    const future_dates = values.dates.split(",").map((d) => d.trim())
    onSubmit({
      future_dates,
      exploration_mode: values.exploration_mode,
      exploration_rate: values.exploration_rate,
    })
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="dates"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Prediction Dates</FormLabel>
              <FormControl>
                <Input {...field} placeholder="YYYY-MM-DD" />
              </FormControl>
              <FormDescription>Enter dates in YYYY-MM-DD format, separated by commas</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="exploration_mode"
          render={({ field }) => (
            <FormItem className="flex items-center justify-between rounded-lg border p-4">
              <div className="space-y-0.5">
                <FormLabel className="text-base">Exploration Mode</FormLabel>
                <FormDescription>Enable to explore new price points</FormDescription>
              </div>
              <FormControl>
                <Switch checked={field.value} onCheckedChange={field.onChange} />
              </FormControl>
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="exploration_rate"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Exploration Rate</FormLabel>
              <FormControl>
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  {...field}
                  onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                />
              </FormControl>
              <FormDescription>Rate of exploration (0-1) for new price points</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <Button type="submit" disabled={isLoading}>
          {isLoading ? "Generating Predictions..." : "Generate Predictions"}
        </Button>
      </form>
    </Form>
  )
}

