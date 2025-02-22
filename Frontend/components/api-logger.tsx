'use client'

import { useEffect, useState } from 'react'
import { Card } from './ui/card'
import { ScrollArea } from './ui/scroll-area'

interface ApiLog {
  timestamp: number
  type: 'request' | 'response' | 'error'
  method: string
  endpoint: string
  data?: any
  duration?: string
  status?: number
}

export function ApiLogger() {
  const [logs, setLogs] = useState<ApiLog[]>([])

  useEffect(() => {
    // Subscribe to API logs
    const handleLog = (event: CustomEvent<ApiLog>) => {
      setLogs(prev => [...prev, event.detail])
    }

    window.addEventListener('api-log' as any, handleLog)
    return () => window.removeEventListener('api-log' as any, handleLog)
  }, [])

  return (
    <Card className="p-4">
      <h3 className="font-semibold mb-2">API Logs</h3>
      <ScrollArea className="h-[300px]">
        {logs.map((log, index) => (
          <div
            key={index}
            className={`mb-2 p-2 rounded text-sm ${
              log.type === 'request'
                ? 'bg-blue-50'
                : log.type === 'response'
                ? 'bg-green-50'
                : 'bg-red-50'
            }`}
          >
            <div className="flex justify-between">
              <span className="font-mono">
                {log.method} {log.endpoint}
              </span>
              <span className="text-gray-500">
                {new Date(log.timestamp).toLocaleTimeString()}
              </span>
            </div>
            {log.duration && (
              <div className="text-gray-500 text-xs">Duration: {log.duration}</div>
            )}
            {log.status && (
              <div className="text-gray-500 text-xs">Status: {log.status}</div>
            )}
            {log.data && (
              <pre className="mt-1 text-xs overflow-x-auto">
                {JSON.stringify(log.data, null, 2)}
              </pre>
            )}
          </div>
        ))}
      </ScrollArea>
    </Card>
  )
}