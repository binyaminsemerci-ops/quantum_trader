import React from 'react'

// API configuration - use environment variables with fallback
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'
const WS_BASE_URL = import.meta.env.VITE_WS_URL || (
  window.location.protocol === 'https:' 
    ? `wss://${window.location.host}` 
    : 'ws://localhost:8000'
)

interface HealthResponse {
  status: string
  message: string
}

interface AIStatus {
  accuracy: number
  sharpe: number
  latency: number
  models: string[]
}

interface Portfolio {
  pnl: number
  exposure: number
  drawdown: number
  positions: number
}

interface Risk {
  var: number
  cvar: number
  volatility: number
  regime: string
}

interface SystemHealth {
  cpu: number
  ram: number
  uptime: number
  containers: number
}

function App() {
  const [health, setHealth] = React.useState<HealthResponse | null>(null)
  const [aiStatus, setAiStatus] = React.useState<AIStatus | null>(null)
  const [portfolio, setPortfolio] = React.useState<Portfolio | null>(null)
  const [risk, setRisk] = React.useState<Risk | null>(null)
  const [systemHealth, setSystemHealth] = React.useState<SystemHealth | null>(null)
  const [wsConnected, setWsConnected] = React.useState(false)

  // WebSocket connection for real-time updates
  React.useEffect(() => {
    // Initial fetch of health endpoint
    fetch(`${API_BASE_URL}/health`)
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error('Health check failed:', err))

    // WebSocket for real-time streaming
    const wsUrl = `${WS_BASE_URL}/stream/live`
    
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => {
      console.log('✅ WebSocket connected')
      setWsConnected(true)
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('📊 WebSocket update:', data)
        
        // Update state with real-time data
        setSystemHealth({
          cpu: data.cpu,
          ram: data.ram,
          uptime: Math.floor(data.timestamp),
          containers: data.containers
        })
        
        setAiStatus({
          accuracy: data.accuracy,
          sharpe: data.sharpe,
          latency: data.latency,
          models: ['XGB', 'LGBM', 'N-HiTS', 'TFT']  // Mock until real data
        })
        
        setPortfolio({
          pnl: data.pnl,
          exposure: data.exposure,
          drawdown: 0.08,  // Mock until real data
          positions: data.positions
        })
        
        // Mock risk data (not in WebSocket yet)
        setRisk({
          var: -0.025,
          cvar: -0.034,
          volatility: 0.18,
          regime: 'Bullish'
        })
      } catch (err) {
        console.error('Error parsing WebSocket message:', err)
      }
    }
    
    ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error)
      setWsConnected(false)
    }
    
    ws.onclose = () => {
      console.log('⚠️ WebSocket disconnected')
      setWsConnected(false)
    }
    
    return () => {
      ws.close()
    }
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold text-white mb-4">
            🧠 Quantum Trader Dashboard
          </h1>
          <p className="text-2xl text-blue-300">
            Real-Time AI Hedge Fund Metrics {wsConnected && <span className="text-green-400">● LIVE</span>}
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {/* System Status */}
          <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
            <h2 className="text-xl font-semibold text-white mb-4">🟢 System Status</h2>
            {health ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className="text-green-400 font-bold">{health.status}</span>
                </div>
                <p className="text-sm text-gray-500 mt-4">{health.message}</p>
              </div>
            ) : (
              <div className="text-yellow-400">Connecting...</div>
            )}
          </div>

          {/* AI Performance */}
          {aiStatus && (
            <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4">🤖 AI Performance</h2>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Accuracy:</span>
                  <span className="text-blue-400 font-bold">{(aiStatus.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Sharpe Ratio:</span>
                  <span className="text-blue-400 font-bold">{aiStatus.sharpe.toFixed(2)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Latency:</span>
                  <span className="text-blue-400 font-bold">{aiStatus.latency}ms</span>
                </div>
                <p className="text-sm text-gray-500 mt-4">Models: {aiStatus.models.join(', ')}</p>
              </div>
            </div>
          )}

          {/* Portfolio */}
          {portfolio && (
            <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4">💼 Portfolio</h2>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">PnL:</span>
                  <span className={`font-bold ${portfolio.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    ${portfolio.pnl.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Exposure:</span>
                  <span className="text-blue-400 font-bold">{(portfolio.exposure * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Drawdown:</span>
                  <span className="text-yellow-400 font-bold">{(portfolio.drawdown * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Positions:</span>
                  <span className="text-blue-400 font-bold">{portfolio.positions}</span>
                </div>
              </div>
            </div>
          )}

          {/* Risk Metrics */}
          {risk && (
            <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4">⚠️ Risk Metrics</h2>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">VaR (95%):</span>
                  <span className="text-red-400 font-bold">{(risk.var * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">CVaR:</span>
                  <span className="text-red-400 font-bold">{(risk.cvar * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Volatility:</span>
                  <span className="text-yellow-400 font-bold">{(risk.volatility * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Regime:</span>
                  <span className="text-blue-400 font-bold">{risk.regime}</span>
                </div>
              </div>
            </div>
          )}

          {/* System Health */}
          {systemHealth && (
            <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4">🖥️ System Health</h2>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">CPU:</span>
                  <span className="text-blue-400 font-bold">{systemHealth.cpu.toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">RAM:</span>
                  <span className="text-blue-400 font-bold">{systemHealth.ram.toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Uptime:</span>
                  <span className="text-green-400 font-bold">{Math.floor(systemHealth.uptime / 3600)}h</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Containers:</span>
                  <span className="text-blue-400 font-bold">{systemHealth.containers}</span>
                </div>
              </div>
            </div>
          )}

          <div className="mt-12 text-gray-400">
            <p>Phase 1 Complete – Base structure operational</p>
            <p className="text-sm mt-2">Next Phase: Database Integration</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
