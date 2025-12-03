'use client'

import { useState } from 'react'
import dynamic from 'next/dynamic'
import { Send, Code2, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

// Dynamically import Monaco Editor to avoid SSR issues
const MonacoEditor = dynamic(() => import('@monaco-editor/react'), {
  ssr: false,
})

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
}) as any

interface GenerateResponse {
  generated_code: string
  output_type: 'plot' | 'text'
  output_data: string
  stdout: string
}

interface ChatMessage {
  id: string
  prompt: string
  response?: GenerateResponse
  timestamp: Date
}

export default function Home() {
  const [prompt, setPrompt] = useState('')
  const [code, setCode] = useState('# Generated code will appear here...\n')
  const [output, setOutput] = useState<{ type: 'plot' | 'text', data: any } | null>(null)
  const [loading, setLoading] = useState(false)
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [selectedChat, setSelectedChat] = useState<string | null>(null)

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setLoading(true)
    const chatId = Date.now().toString()
    const newMessage: ChatMessage = {
      id: chatId,
      prompt: prompt,
      timestamp: new Date(),
    }

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      
      if (!API_URL || API_URL === 'http://localhost:8000') {
        throw new Error('API URL not configured. Please set NEXT_PUBLIC_API_URL environment variable in Vercel.')
      }
      
      // Remove trailing slash if present
      const cleanApiUrl = API_URL.replace(/\/$/, '')
      const apiEndpoint = `${cleanApiUrl}/api/generate`
      
      console.log('Calling API:', apiEndpoint)
      console.log('API URL from env:', API_URL)
      
      // Create abort controller for timeout (Render free tier can be slow to wake up)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 60000) // 60 second timeout
      
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
        signal: controller.signal,
      })
      
      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorText = await response.text()
        let errorMsg = `HTTP ${response.status} error`
        
        if (response.status === 404) {
          errorMsg = `404 Not Found - Backend endpoint not found. Check: 1) Backend URL is correct (${cleanApiUrl}), 2) Backend is running, 3) Try accessing ${cleanApiUrl}/health to verify backend is up.`
        } else {
          errorMsg = `HTTP error! status: ${response.status} - ${errorText}`
        }
        
        throw new Error(errorMsg)
      }

      const data: GenerateResponse = await response.json()
      
      newMessage.response = data
      setChatHistory(prev => [newMessage, ...prev])
      setSelectedChat(chatId)
      
      setCode(data.generated_code)
      
      if (data.output_type === 'plot') {
        try {
          const plotData = JSON.parse(data.output_data)
          setOutput({ type: 'plot', data: plotData })
        } catch (e) {
          setOutput({ type: 'text', data: `Error parsing plot data: ${e}` })
        }
      } else {
        setOutput({ type: 'text', data: data.output_data || data.stdout || 'No output' })
      }
    } catch (error) {
      console.error('Error:', error)
      let errorMessage = 'Unknown error'
      
      if (error instanceof Error) {
        if (error.name === 'AbortError' || error.message.includes('timeout')) {
          errorMessage = 'Request timed out. The backend may be sleeping (Render free tier). Please try again - it may take 30-60 seconds to wake up.'
        } else if (error.message.includes('Failed to fetch') || error.message.includes('Load failed') || error.message.includes('404')) {
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'API URL not set'
          errorMessage = `Failed to connect to backend at ${apiUrl}. Troubleshooting: 1) Verify backend is running at ${apiUrl}/health, 2) Check CORS is configured, 3) Verify NEXT_PUBLIC_API_URL is set in Vercel environment variables, 4) Backend may be sleeping (Render free tier) - wait 30-60 seconds and try again.`
        } else {
          errorMessage = error.message
        }
      }
      
      setOutput({ type: 'text', data: `Error: ${errorMessage}` })
    } finally {
      setLoading(false)
      setPrompt('')
    }
  }

  const loadChat = (chat: ChatMessage) => {
    setSelectedChat(chat.id)
    setPrompt(chat.prompt)
    if (chat.response) {
      setCode(chat.response.generated_code)
      if (chat.response.output_type === 'plot') {
        try {
          const plotData = JSON.parse(chat.response.output_data)
          setOutput({ type: 'plot', data: plotData })
        } catch (e) {
          setOutput({ type: 'text', data: chat.response.output_data })
        }
      } else {
        setOutput({ type: 'text', data: chat.response.output_data || chat.response.stdout || 'No output' })
      }
    }
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Sidebar - Chat History */}
      <div className="w-64 border-r border-border bg-card flex flex-col">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold text-primary flex items-center gap-2">
            <Code2 className="w-5 h-5" />
            Basis
          </h2>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {chatHistory.length === 0 ? (
            <p className="text-sm text-muted-foreground p-4 text-center">
              No chat history yet
            </p>
          ) : (
            <div className="space-y-2">
              {chatHistory.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => loadChat(chat)}
                  className={`w-full text-left p-3 rounded-md text-sm transition-colors ${
                    selectedChat === chat.id
                      ? 'bg-primary/20 border border-primary/50'
                      : 'bg-secondary hover:bg-accent'
                  }`}
                >
                  <p className="truncate text-foreground">{chat.prompt}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {chat.timestamp.toLocaleTimeString()}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar - Input */}
        <div className="border-b border-border p-4 bg-card">
          <div className="flex gap-2">
            <Input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleGenerate()
                }
              }}
              placeholder="Ask a financial question... (e.g., 'Show me AAPL stock price over the last 6 months')"
              className="flex-1"
              disabled={loading}
            />
            <Button
              onClick={handleGenerate}
              disabled={loading || !prompt.trim()}
              className="min-w-[100px]"
            >
              {loading ? (
                'Loading...'
              ) : (
                <>
                  <Send className="w-4 h-4 mr-2" />
                  Generate
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Split Pane - Code Editor and Preview */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left Pane - Monaco Editor */}
          <div className="w-1/2 border-r border-border flex flex-col">
            <div className="p-2 border-b border-border bg-card flex items-center gap-2">
              <Code2 className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">Generated Code</span>
            </div>
            <div className="flex-1">
              <MonacoEditor
                height="100%"
                defaultLanguage="python"
                theme="vs-dark"
                value={code}
                onChange={(value) => setCode(value || '')}
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  readOnly: false,
                }}
              />
            </div>
          </div>

          {/* Right Pane - Preview */}
          <div className="w-1/2 flex flex-col">
            <div className="p-2 border-b border-border bg-card flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">Preview</span>
            </div>
            <div className="flex-1 overflow-auto p-4 bg-background">
              {output ? (
                output.type === 'plot' ? (
                  <div className="h-full w-full min-h-[500px]">
                    {output.data && output.data.data ? (
                      <Plot
                        data={output.data.data.map((trace: any) => {
                          // Flatten y-axis data if nested (safety check)
                          if (trace.y && Array.isArray(trace.y) && trace.y.length > 0) {
                            if (Array.isArray(trace.y[0])) {
                              // Nested array, flatten it
                              trace.y = trace.y.map((item: any) => 
                                Array.isArray(item) ? (item.length > 0 ? item[0] : item) : item
                              )
                            }
                            // Ensure all y values are numbers
                            trace.y = trace.y.map((y: any) => {
                              if (Array.isArray(y)) {
                                return y.length > 0 ? Number(y[0]) : 0
                              }
                              return Number(y) || 0
                            })
                          }
                          
                          // Ensure x-axis data is properly formatted for dates
                          if (trace.x && Array.isArray(trace.x) && trace.x.length > 0) {
                            const firstX = trace.x[0]
                            // Convert large numbers or string timestamps (nanosecond timestamps) to date strings
                            if (typeof firstX === 'number' && firstX > 1e15) {
                              // Numeric nanosecond timestamp
                              trace.x = trace.x.map((x: any) => {
                                try {
                                  // Convert nanoseconds to ISO string
                                  const date = new Date(Number(x) / 1e6) // Convert ns to ms
                                  return date.toISOString()
                                } catch {
                                  return String(x)
                                }
                              })
                            } else if (typeof firstX === 'string') {
                              // Check if it's a string representation of a large number (nanosecond timestamp)
                              const firstXNum = parseFloat(firstX)
                              if (!isNaN(firstXNum) && firstXNum > 1e15) {
                                // String representation of nanosecond timestamp
                                trace.x = trace.x.map((x: any) => {
                                  try {
                                    const num = parseFloat(String(x))
                                    if (!isNaN(num) && num > 1e15) {
                                      // Convert nanoseconds to ISO string
                                      const date = new Date(num / 1e6) // Convert ns to ms
                                      return date.toISOString()
                                    }
                                    return String(x)
                                  } catch {
                                    return String(x)
                                  }
                                })
                              } else if (firstX.includes('T') || firstX.match(/^\d{4}-\d{2}-\d{2}/)) {
                                // Already ISO format or date-like string, ensure all are strings
                                trace.x = trace.x.map((x: any) => typeof x === 'string' ? x : new Date(x).toISOString())
                              }
                            }
                          }
                          return trace
                        })}
                        layout={{
                          ...(output.data.layout || {}),
                          paper_bgcolor: 'rgba(0,0,0,0)',
                          plot_bgcolor: 'rgba(0,0,0,0)',
                          font: { color: '#e5e7eb' },
                          xaxis: {
                            ...(output.data.layout?.xaxis || {}),
                            type: (() => {
                              const firstTrace = output.data.data?.[0]
                              if (firstTrace?.x && Array.isArray(firstTrace.x) && firstTrace.x.length > 0) {
                                const firstX = firstTrace.x[0]
                                if (typeof firstX === 'string') {
                                  // Check if it's a date string or a large number string
                                  if (firstX.includes('T') || firstX.match(/^\d{4}-\d{2}-\d{2}/)) {
                                    return 'date'
                                  }
                                  const firstXNum = parseFloat(firstX)
                                  if (!isNaN(firstXNum) && firstXNum > 1e15) {
                                    return 'date'
                                  }
                                }
                                if (typeof firstX === 'number' && firstX > 1e15) {
                                  return 'date'
                                }
                              }
                              return output.data.layout?.xaxis?.type || 'linear'
                            })(),
                            gridcolor: 'rgba(255,255,255,0.1)',
                          },
                          yaxis: {
                            ...(output.data.layout?.yaxis || {}),
                            gridcolor: 'rgba(255,255,255,0.1)',
                          },
                        }}
                        config={{
                          ...(output.data.config || {}),
                          displayModeBar: true,
                          displaylogo: false,
                          modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                        }}
                        style={{ width: '100%', height: '100%', minHeight: '500px' }}
                        useResizeHandler={true}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full text-muted-foreground">
                        <p>Error: Invalid plot data structure</p>
                        <pre className="text-xs mt-2">{JSON.stringify(output.data, null, 2).substring(0, 200)}</pre>
                      </div>
                    )}
                  </div>
                ) : (
                  <Card>
                    <CardHeader>
                      <CardTitle>Output</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-sm font-mono whitespace-pre-wrap text-foreground">
                        {output.data}
                      </pre>
                    </CardContent>
                  </Card>
                )
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  <p>No output yet. Generate code to see results here.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

