import React, { useState, useEffect } from 'react';
import { TrendingUp, Activity, DollarSign, BarChart3, Eye, Filter, Search, Layers, Shield, Coins, Gamepad2, Zap, Database, Banknote } from 'lucide-react';

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

interface MarketData {
  market_status: string;
  volatility: number;
  sentiment: string;
  volume_24h: string;
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

const ProfessionalDashboard: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [aiStatus, setAiStatus] = useState<AITradingStatus | null>(null);
  const [signals, setSignals] = useState<SignalData[]>([]);
  const [learningStatus, setLearningStatus] = useState<LearningStatus | null>(null);
  const [watchlist, setWatchlist] = useState<CoinData[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Get unique categories from watchlist
  const categories = ['All', ...Array.from(new Set(watchlist.map(coin => coin.category)))];

  // Filter coins based on category and search
  const filteredCoins = watchlist.filter(coin => {
    const matchesCategory = selectedCategory === 'All' || coin.category === selectedCategory;
    const matchesSearch = coin.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                         coin.symbol.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const fetchData = async () => {
    try {
      const [systemRes, portfolioRes, marketRes, aiRes, signalsRes, learningRes, watchlistRes] = await Promise.all([
        fetch('http://localhost:8000/api/v1/system/status'),
        fetch('http://localhost:8000/api/v1/portfolio'),
        fetch('http://localhost:8000/api/v1/portfolio/market-overview'),
        fetch('http://localhost:8000/api/v1/ai-trading/status'),
        fetch('http://localhost:8000/api/v1/signals/recent?limit=5'),
        fetch('http://localhost:8000/api/v1/continuous-learning/status'),
        fetch('http://localhost:8000/api/v1/watchlist')
      ]);

      if (systemRes.ok) setSystemStatus(await systemRes.json());
      if (portfolioRes.ok) setPortfolio(await portfolioRes.json());
      if (marketRes.ok) setMarketData(await marketRes.json());
      if (aiRes.ok) setAiStatus(await aiRes.json());
      if (signalsRes.ok) setSignals(await signalsRes.json());
      if (learningRes.ok) setLearningStatus(await learningRes.json());
      if (watchlistRes.ok) setWatchlist(await watchlistRes.json());
    } catch (error) {
      console.error('Data fetch error:', error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, []);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Layer 1': return <Layers className="w-4 h-4" />;
      case 'Layer 2': return <Zap className="w-4 h-4" />;
      case 'DeFi': return <Coins className="w-4 h-4" />;
      case 'Gaming': return <Gamepad2 className="w-4 h-4" />;
      case 'Privacy': return <Shield className="w-4 h-4" />;
      case 'Storage': return <Database className="w-4 h-4" />;
      case 'Stablecoin': return <Banknote className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const MiniSparkline = ({ data }: { data: number[] }) => {
    if (!data || data.length < 2) return null;
    
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 60;
      const y = 20 - ((value - min) / range) * 15;
      return `${x},${y}`;
    }).join(' ');
    
    const isPositive = data[data.length - 1] > data[0];
    
    return (
      <svg width="60" height="20" className="inline-block">
        <polyline
          points={points}
          fill="none"
          stroke={isPositive ? "#10b981" : "#ef4444"}
          strokeWidth="1.5"
          className="opacity-80"
        />
      </svg>
    );
  };

  const formatPrice = (price: number) => {
    if (price < 0.01) return price.toFixed(8);
    if (price < 1) return price.toFixed(6);
    if (price < 100) return price.toFixed(4);
    return price.toFixed(2);
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1e9) return `$${(volume / 1e9).toFixed(2)}B`;
    if (volume >= 1e6) return `$${(volume / 1e6).toFixed(2)}M`;
    if (volume >= 1e3) return `$${(volume / 1e3).toFixed(2)}K`;
    return `$${volume.toFixed(2)}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="text-center py-6">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-green-400 bg-clip-text text-transparent">
            QUANTUM TRADER PRO
          </h1>
          <p className="text-slate-400 mt-2">Professional AI Trading Dashboard</p>
        </div>

        {/* Top Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          
          {/* AI Trading Engine */}
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-2xl p-6 hover:bg-slate-800/70 transition-all">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-200">AI Engine</h3>
              <BarChart3 className="w-6 h-6 text-blue-400" />
            </div>
            {aiStatus ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-400">Status:</span>
                  <span className="text-green-400 font-semibold">{aiStatus.engine_status}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Confidence:</span>
                  <span className="text-blue-400 font-semibold">{(aiStatus.confidence_level * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Win Rate:</span>
                  <span className="text-green-400 font-semibold">{aiStatus.win_rate.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Trades Today:</span>
                  <span className="text-white font-semibold">{aiStatus.trades_today}</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500">Loading AI data...</div>
            )}
          </div>

          {/* Learning Status */}
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-2xl p-6 hover:bg-slate-800/70 transition-all">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-200">Learning</h3>
              <Activity className="w-6 h-6 text-purple-400" />
            </div>
            {learningStatus ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-400">Status:</span>
                  <span className={`font-semibold ${learningStatus.learning_active ? 'text-green-400' : 'text-red-400'}`}>
                    {learningStatus.learning_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Accuracy:</span>
                  <span className="text-blue-400 font-semibold">{(learningStatus.model_accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Data Points:</span>
                  <span className="text-white font-semibold">{learningStatus.data_points.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Symbols:</span>
                  <span className="text-white font-semibold">{learningStatus.symbols.length}</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500">Loading learning data...</div>
            )}
          </div>

          {/* Portfolio */}
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-2xl p-6 hover:bg-slate-800/70 transition-all">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-200">Portfolio</h3>
              <DollarSign className="w-6 h-6 text-green-400" />
            </div>
            {portfolio ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-400">Balance:</span>
                  <span className="text-white font-bold text-xl">${portfolio.balance.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">24h P&L:</span>
                  <span className={`font-semibold ${portfolio.pnl_24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {portfolio.pnl_24h >= 0 ? '+' : ''}${portfolio.pnl_24h.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Positions:</span>
                  <span className="text-white font-semibold">{portfolio.positions}</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500">Loading portfolio...</div>
            )}
          </div>

          {/* System Status */}
          <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-2xl p-6 hover:bg-slate-800/70 transition-all">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-200">System</h3>
              <Eye className="w-6 h-6 text-orange-400" />
            </div>
            {systemStatus ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-400">Status:</span>
                  <span className="text-green-400 font-semibold">{systemStatus.status}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Uptime:</span>
                  <span className="text-white font-semibold">{systemStatus.uptime}h</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">CPU:</span>
                  <span className="text-blue-400 font-semibold">{systemStatus.cpu_usage}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Memory:</span>
                  <span className="text-purple-400 font-semibold">{systemStatus.memory_usage}%</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500">Loading system data...</div>
            )}
          </div>
        </div>

        {/* AI Signals Section */}
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-2xl p-6">
          <h3 className="text-xl font-semibold text-slate-200 mb-6 flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-green-400" />
            AI Trading Signals
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {signals.map((signal, index) => (
              <div key={index} className="bg-slate-700/50 rounded-xl p-4 border border-slate-600/50 hover:bg-slate-700/70 transition-all">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-lg">{signal.symbol}</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                      signal.action === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {signal.action}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-slate-400">Confidence</div>
                    <div className="text-lg font-bold text-blue-400">{signal.confidence}%</div>
                  </div>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Price:</span>
                    <span className="text-white font-semibold">${formatPrice(signal.price)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Target:</span>
                    <span className="text-green-400 font-semibold">${formatPrice(signal.price_prediction.target)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Probability:</span>
                    <span className="text-blue-400 font-semibold">{signal.price_prediction.probability}%</span>
                  </div>
                  <div className="mt-3 p-2 bg-slate-800/50 rounded text-xs text-slate-300">
                    <strong>AI Reasoning:</strong> {signal.reasoning}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Comprehensive Watchlist */}
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-2xl p-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4">
            <h3 className="text-xl font-semibold text-slate-200 flex items-center gap-3">
              <Coins className="w-6 h-6 text-yellow-400" />
              Market Watchlist ({watchlist.length} Coins)
            </h3>
            
            {/* Filters and Search */}
            <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search coins..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="bg-slate-700/50 border border-slate-600/50 rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:border-blue-400/50 transition-colors"
                />
              </div>
              <div className="relative">
                <Filter className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  title="Filter by category"
                  className="bg-slate-700/50 border border-slate-600/50 rounded-lg pl-10 pr-8 py-2 text-sm focus:outline-none focus:border-blue-400/50 transition-colors appearance-none"
                >
                  {categories.map(category => (
                    <option key={category} value={category}>{category}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Coins Table */}
          <div className="overflow-x-auto">
            <div className="grid grid-cols-1 gap-3 max-h-96 overflow-y-auto">
              {filteredCoins.map((coin) => (
                <div key={coin.symbol} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/30 hover:bg-slate-700/50 transition-all">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                          {getCategoryIcon(coin.category)}
                          <span className="text-xs text-slate-400 uppercase">{coin.category}</span>
                        </div>
                        <div>
                          <div className="font-bold text-lg">{coin.symbol}</div>
                          <div className="text-sm text-slate-400">{coin.name}</div>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-lg font-bold">${formatPrice(coin.price)}</div>
                        <div className={`text-sm font-semibold ${coin.change24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {coin.change24h >= 0 ? '+' : ''}{coin.change24h.toFixed(2)}%
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm text-slate-400">Volume 24h</div>
                        <div className="text-sm font-semibold">{formatVolume(coin.volume24h)}</div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm text-slate-400">AI Confidence</div>
                        <div className="text-sm font-bold text-blue-400">{(coin.confidence * 100).toFixed(1)}%</div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm text-slate-400 mb-1">Trend</div>
                        <MiniSparkline data={coin.sparkline} />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {filteredCoins.length === 0 && (
            <div className="text-center py-8 text-slate-400">
              No coins found matching your filters
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default ProfessionalDashboard;