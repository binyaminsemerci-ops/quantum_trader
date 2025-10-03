import React, { useState, useEffect } from 'react';
import { TrendingUp, Activity, DollarSign, BarChart3, Eye, Search, Plus, X, Zap, Brain, Target, TrendingDown } from 'lucide-react';

interface CoinData {
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  volume24h: number;
  category: string;
  confidence: number;
  sparkline: number[];
  ts: string;
}

interface SystemStatus {
  status: string;
  uptime: number;
  cpu_usage: number;
  memory_usage: number;
  connections: number;
}

interface PortfolioData {
  balance: number;
  pnl_24h: number;
  positions: number;
  timestamp: string;
}

interface AITradingStatus {
  engine_status: string;
  confidence_level: number;
  trades_today: number;
  win_rate: number;
  learning_active: boolean;
  symbols_monitored: number;
  last_signal_time: string;
}

interface SignalData {
  symbol: string;
  action: string;
  confidence: number;
  price: number;
  reasoning: string;
  timestamp: string;
  price_prediction: {
    target: number;
    timeframe: string;
    probability: number;
  };
}

interface LearningStatus {
  learning_active: boolean;
  data_points: number;
  model_accuracy: number;
  enabled: boolean;
  symbols: string[];
  total_signals: number;
  continuous_learning_status: string;
}

interface Candle {
  time?: string; // from /prices/recent demo path
  timestamp?: string; // fallback for /candles
  open?: number; high?: number; low?: number; close?: number; volume?: number;
}

const FullSizeDashboard: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [aiStatus, setAiStatus] = useState<AITradingStatus | null>(null);
  const [signals, setSignals] = useState<SignalData[]>([]);
  const [learningStatus, setLearningStatus] = useState<LearningStatus | null>(null);
  const [allCoins, setAllCoins] = useState<CoinData[]>([]);
  const [selectedCoins, setSelectedCoins] = useState<CoinData[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [showSearch, setShowSearch] = useState<boolean>(false);
  const [activeChartSymbol, setActiveChartSymbol] = useState<string | null>(null);
  const [chartData, setChartData] = useState<Candle[]>([]);
  const [chartLoading, setChartLoading] = useState<boolean>(false);

  // Defensive null-safe accessors
  const safeSystemStatus = systemStatus || { status: 'LOADING', uptime: 0, cpu_usage: 0, memory_usage: 0, connections: 0 };
  const safePortfolio = portfolio || { balance: 0, pnl_24h: 0, positions: 0, timestamp: '' };
  const safeAiStatus = aiStatus || { confidence_level: 0, win_rate: 0, engine_status: 'Ready', trades_today: 0 };
  const safeLearningStatus = learningStatus || { learning_active: false, data_points: 0, model_accuracy: 0, enabled: false, symbols: [], total_signals: 0, continuous_learning_status: 'inactive' };
  const safeSignals = signals || [];
  const safeAllCoins = allCoins || [];
  const safeSelectedCoins = selectedCoins || [];

  const fetchChart = async (symbol: string) => {
    try {
      setChartLoading(true);
      // Try fast lightweight synthetic/live path first
      const res = await fetch(`http://localhost:8000/api/v1/prices/recent?symbol=${symbol}&limit=60`);
      if (res.ok) {
        const candles = await res.json();
        setChartData(candles);
      } else {
        // fallback to /candles (DB or demo)
        const res2 = await fetch(`http://localhost:8000/api/v1/candles?symbol=${symbol}&limit=60`);
        if (res2.ok) {
          const json = await res2.json();
            setChartData(json.candles || []);
        }
      }
    } catch (e) {
      console.warn('chart fetch error', e);
    } finally {
      setChartLoading(false);
    }
  };

  // Filter coins for search
  const searchResults = allCoins.filter(coin => 
    !selectedCoins.some(selected => selected.symbol === coin.symbol) &&
    ((coin.name || '').toLowerCase().includes(searchTerm.toLowerCase()) || 
     (coin.symbol || '').toLowerCase().includes(searchTerm.toLowerCase()))
  ).slice(0, 20);

  const fetchData = async () => {
    try {
      const [systemRes, portfolioRes, aiRes, signalsRes, learningRes, watchlistRes] = await Promise.all([
        fetch('http://localhost:8000/api/v1/system/status'),
        fetch('http://localhost:8000/api/v1/portfolio'),
        fetch('http://localhost:8000/api/v1/ai-trading/status'),
        fetch('http://localhost:8000/api/v1/signals/recent?limit=5'),
        fetch('http://localhost:8000/api/v1/continuous-learning/status'),
        fetch('http://localhost:8000/api/v1/watchlist/full')
      ]);

      if (systemRes.ok) {
        // Map backend system status (service, uptime_seconds, evaluator_running, etc.)
        // to the legacy UI's expected shape (status, uptime (hrs), cpu_usage, memory_usage, connections)
        const raw = await systemRes.json();
        const mapped: SystemStatus = {
          status: raw.evaluator_running ? 'RUNNING' : 'IDLE',
          uptime: raw.uptime_seconds ? Math.floor(raw.uptime_seconds / 3600) : 0,
            // Backend doesn't yet expose these – keep placeholders so UI stays stable
          cpu_usage: typeof raw.cpu_usage === 'number' ? raw.cpu_usage : 0,
          memory_usage: typeof raw.memory_usage === 'number' ? raw.memory_usage : 0,
          connections: typeof raw.connections === 'number' ? raw.connections : 0,
        };
        setSystemStatus(mapped);
      }
      if (portfolioRes.ok) {
        const rawPort = await portfolioRes.json();
        // Backend returns { totalValue, totalPnL, totalPnLPercent, positions: [ ... ] }
        // Map to legacy UI shape (balance, pnl_24h, positions (count))
        const mappedPort: PortfolioData = {
          balance: typeof rawPort.totalValue === 'number' ? rawPort.totalValue : 0,
          pnl_24h: typeof rawPort.totalPnL === 'number' ? rawPort.totalPnL : 0,
          positions: Array.isArray(rawPort.positions) ? rawPort.positions.length : 0,
          timestamp: new Date().toISOString(),
        };
        setPortfolio(mappedPort);
      }
      if (aiRes.ok) setAiStatus(await aiRes.json());
      if (signalsRes.ok) setSignals(await signalsRes.json());
      if (learningRes.ok) setLearningStatus(await learningRes.json());
      if (watchlistRes.ok) {
        const coinData = await watchlistRes.json();
        setAllCoins(coinData);
        if (selectedCoins.length === 0) {
            setSelectedCoins(coinData.slice(0, 10));
        }
      } else if (watchlistRes.status === 404) {
        // fallback to DB watchlist minimal list
        const fallback = await fetch('http://localhost:8000/api/v1/watchlist');
        if (fallback.ok) {
          const baseList = await fallback.json();
          const mapped = (baseList || []).map((r: any) => ({
            symbol: r.symbol,
            name: r.symbol,
            price: 0,
            change24h: 0,
            volume24h: 0,
            category: 'Unknown',
            confidence: 0,
            sparkline: [],
            ts: new Date().toISOString()
          }));
          setAllCoins(mapped);
          if (selectedCoins.length === 0) {
              setSelectedCoins(mapped.slice(0, 10));
          }
        }
      }
    } catch (error) {
      console.error('Data fetch error:', error);
    }
  };

  useEffect(() => {
    // On first mount, attempt to hydrate selectedCoins from localStorage
    if (selectedCoins.length === 0) {
      try {
        const raw = localStorage.getItem('qt.selectedCoins');
        if (raw) {
          const parsed: CoinData[] = JSON.parse(raw);
          if (Array.isArray(parsed) && parsed.length > 0) {
            setSelectedCoins(parsed.slice(0, 10));
          }
        }
      } catch (e) {
        console.warn('LocalStorage hydrate failed', e);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 5000); // Reduced from 3000ms to 5000ms for stability
    return () => clearInterval(interval);
  }, [selectedCoins.length]);

  // Persist whenever selectedCoins changes
  useEffect(() => {
    try {
      localStorage.setItem('qt.selectedCoins', JSON.stringify(selectedCoins));
    } catch (e) {
      // non-fatal
    }
  }, [selectedCoins]);

  const addCoin = (coin: CoinData) => {
    if (selectedCoins.length >= 10) {
      // Replace the last coin
      setSelectedCoins([...selectedCoins.slice(0, 9), coin]);
    } else {
      setSelectedCoins([...selectedCoins, coin]);
    }
    setSearchTerm('');
    setShowSearch(false);
  };

  const removeCoin = (symbolToRemove: string) => {
    setSelectedCoins(selectedCoins.filter(coin => coin.symbol !== symbolToRemove));
    if (activeChartSymbol === symbolToRemove) {
      setActiveChartSymbol(null);
      setChartData([]);
    }
  };

  const MiniSparkline = ({ data }: { data: number[] }) => {
    if (!data || data.length < 2) return null;
    
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 40;
      const y = 15 - ((value - min) / range) * 10;
      return `${x},${y}`;
    }).join(' ');
    
    const isPositive = data[data.length - 1] > data[0];
    
    return (
      <svg width="40" height="15" className="inline-block">
        <polyline
          points={points}
          fill="none"
          stroke={isPositive ? "#10b981" : "#ef4444"}
          strokeWidth="1"
        />
      </svg>
    );
  };

  // HELHETLIG SAFE toFixed som ALDRI krasjer
  const safeToFixed = (value: any, decimals: number = 2): string => {
    const num = Number(value);
    if (isNaN(num) || value === undefined || value === null) return '0.' + '0'.repeat(decimals);
    return num.toFixed(decimals);
  };

  const formatPrice = (price: number | undefined) => {
    const num = Number(price);
    if (isNaN(num) || price === undefined || price === null) return '0.00';
    if (num < 0.01) return safeToFixed(num, 8);
    if (num < 1) return safeToFixed(num, 6);
    if (num < 100) return safeToFixed(num, 4);
    return safeToFixed(num, 2);
  };

  // Defensive: avoid white screen if something is fundamentally missing
  if (!allCoins && !systemStatus && !portfolio && !aiStatus && !learningStatus) {
    return <div className="min-h-screen flex items-center justify-center bg-gray-900 text-gray-300">Laster...</div>;
  }

  return (
    <div className="h-screen bg-gray-900 text-white overflow-hidden">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          QUANTUM TRADER DASHBOARD
        </h1>
      </div>

      {/* Main Layout */}
      <div className="h-[calc(100vh-80px)] grid grid-cols-12 gap-4 p-4">
        
        {/* Left Column - Stats */}
        <div className="col-span-3 space-y-4">
          
          {/* System Status */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Eye className="w-5 h-5 text-green-400" />
              <h3 className="font-semibold">System</h3>
            </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className="text-green-400">{safeSystemStatus.status}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Uptime:</span>
                  <span>{safeSystemStatus.uptime}h</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">CPU:</span>
                  <span className="text-blue-400">{safeSystemStatus.cpu_usage}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Memory:</span>
                  <span className="text-purple-400">{safeSystemStatus.memory_usage}%</span>
                </div>
              </div>
          </div>

          {/* Portfolio */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <DollarSign className="w-5 h-5 text-green-400" />
              <h3 className="font-semibold">Portfolio</h3>
            </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Balance:</span>
                  <span className="font-bold text-lg">${safePortfolio.balance.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">24h P&L:</span>
                  <span className={`font-semibold ${safePortfolio.pnl_24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {safePortfolio.pnl_24h >= 0 ? '+' : ''}${safeToFixed(safePortfolio.pnl_24h, 2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Positions:</span>
                  <span>{safePortfolio.positions}</span>
                </div>
              </div>
          </div>

          {/* AI Engine */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="w-5 h-5 text-blue-400" />
              <h3 className="font-semibold">AI Engine</h3>
            </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className="text-green-400">{safeAiStatus.engine_status}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Confidence:</span>
                  <span className="text-blue-400">{safeToFixed(safeAiStatus.confidence_level * 100, 1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Win Rate:</span>
                  <span className="text-green-400">{safeToFixed(safeAiStatus.win_rate, 1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Trades:</span>
                  <span>{safeAiStatus.trades_today}</span>
                </div>
              </div>
          </div>

          {/* Learning Status */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-5 h-5 text-purple-400" />
              <h3 className="font-semibold">Learning</h3>
            </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Active:</span>
                  <span className={safeLearningStatus.learning_active ? 'text-green-400' : 'text-red-400'}>
                    {safeLearningStatus.learning_active ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Accuracy:</span>
                  <span className="text-blue-400">{safeToFixed(safeLearningStatus.model_accuracy * 100, 1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Data Points:</span>
                  <span>{safeLearningStatus.data_points.toLocaleString()}</span>
                </div>
              </div>
          </div>
        </div>

        {/* Center Column - Signals and Charts */}
        <div className="col-span-6 space-y-4">
          
          {/* AI Signals */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 h-64">
            <div className="flex items-center gap-2 mb-4">
              <Target className="w-5 h-5 text-yellow-400" />
              <h3 className="font-semibold">AI Trading Signals</h3>
            </div>
            <div className="space-y-3 h-48 overflow-y-auto">
              {safeSignals.map((signal, index) => (
                <div key={index} className="bg-gray-700 rounded-lg p-3 border border-gray-600">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-bold">{signal.symbol}</span>
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        signal.action === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                        {signal.action}
                      </span>
                    </div>
                    <div className="text-blue-400 font-semibold">{signal.confidence}%</div>
                  </div>
                  <div className="text-xs text-gray-400 mb-2">
                    Target: ${formatPrice(signal.price_prediction?.target || 0)} | 
                    Probability: {signal.price_prediction?.probability || 0}%
                  </div>
                  <div className="text-xs text-gray-300 bg-gray-800 p-2 rounded">
                    {signal.reasoning}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chart Placeholder */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 flex-1">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-400" />
                <h3 className="font-semibold">Price Chart</h3>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-400">
                {activeChartSymbol ? <span>{activeChartSymbol}</span> : <span>Select coin</span>}
              </div>
            </div>
            <div className="h-64 bg-gray-700 rounded-lg flex items-center justify-center relative">
              {activeChartSymbol && chartData.length > 2 ? (
                <ChartCanvas data={chartData} />
              ) : (
                <div className="text-center text-gray-400">
                  <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <div>{activeChartSymbol ? 'No chart data' : 'Click a watchlist coin for chart'}</div>
                </div>
              )}
              {chartLoading && (
                <div className="absolute top-2 right-2 text-xs text-gray-300 bg-gray-800/70 px-2 py-1 rounded">Loading...</div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column - Coin Watchlist */}
        <div className="col-span-3 space-y-4">
          
          {/* Watchlist Header with Search */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Watchlist ({safeSelectedCoins.length}/10)
              </h3>
              <button
                onClick={() => setShowSearch(!showSearch)}
                title="Add coin to watchlist"
                className="p-1 hover:bg-gray-700 rounded"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>

            {/* Search */}
            {showSearch && (
              <div className="mb-3">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-2 top-2.5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search coins..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target?.value || '')}
                    className="w-full bg-gray-700 border border-gray-600 rounded pl-8 pr-4 py-2 text-sm"
                  />
                </div>
                {searchTerm && (
                  <div className="mt-2 max-h-32 overflow-y-auto bg-gray-700 border border-gray-600 rounded">
                    {searchResults.map((coin) => (
                      <button
                        key={coin.symbol}
                        onClick={() => addCoin(coin)}
                        className="w-full text-left p-2 hover:bg-gray-600 flex justify-between items-center"
                      >
                        <div>
                          <div className="font-semibold text-xs">{coin.symbol}</div>
                          <div className="text-xs text-gray-400">{coin.name}</div>
                        </div>
                        <div className="text-xs">${formatPrice(coin.price || 0)}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Selected Coins */}
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {safeSelectedCoins.map((coin) => (
                <div key={coin.symbol} className={`bg-gray-700 rounded-lg p-3 border ${activeChartSymbol === coin.symbol ? 'border-blue-500' : 'border-gray-600'} cursor-pointer`}
                  onClick={() => {
                    setActiveChartSymbol(coin.symbol);
                    fetchChart(coin.symbol);
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold text-sm">{coin.symbol}</span>
                        <button
                          onClick={() => removeCoin(coin.symbol)}
                          title="Remove coin from watchlist"
                          className="p-1 hover:bg-gray-600 rounded"
                          onMouseDown={(e) => e.stopPropagation()}
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                      <div className="text-xs text-gray-400 mb-1">{coin.name}</div>
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">${formatPrice(coin.price || 0)}</span>
                        <span className={`text-xs font-semibold ${(coin.change24h || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {(coin.change24h || 0) >= 0 ? (
                            <TrendingUp className="w-3 h-3 inline mr-1" />
                          ) : (
                            <TrendingDown className="w-3 h-3 inline mr-1" />
                          )}
                          {safeToFixed(Math.abs(coin.change24h || 0), 2)}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <MiniSparkline data={coin.sparkline} />
                        <span className="text-xs text-blue-400">{safeToFixed((coin.confidence || 0) * 100, 0)}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {safeSelectedCoins.length === 0 && (
              <div className="text-center py-4 text-gray-500 text-sm">
                No coins selected
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FullSizeDashboard;

// Lightweight line chart using inline SVG (no external deps)
const ChartCanvas: React.FC<{ data: Candle[] }> = ({ data }) => {
  const closes = data.map(c => c.close ?? 0);
  if (closes.length < 2) return null;
  const max = Math.max(...closes);
  const min = Math.min(...closes);
  const range = max - min || 1;
  const points = closes.map((v, i) => {
    const x = (i / (closes.length - 1)) * 100;
    const y = 100 - ((v - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');
  const gradientId = 'chartGradient';
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full">
      <defs>
        <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.7} />
          <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.1} />
        </linearGradient>
      </defs>
      {/* Area fill */}
      <polyline
        points={`0,100 ${points} 100,100`}
        fill={`url(#${gradientId})`}
        stroke="none"
      />
      {/* Line */}
      <polyline
        points={points}
        fill="none"
        stroke="#60a5fa"
        strokeWidth={1.2}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
};