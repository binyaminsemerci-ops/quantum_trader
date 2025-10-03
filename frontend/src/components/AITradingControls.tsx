import { useState, useEffect, memo } from 'react';
import { Brain, Pause, Settings, Zap, AlertTriangle } from 'lucide-react';

interface AITradingControlsProps {
  onStartAITrading: (symbols: string[]) => Promise<void>;
  onStopAITrading: () => Promise<void>;
  onUpdateAIConfig: (config: any) => Promise<void>;
  aiTradingStatus: 'active' | 'inactive' | 'unknown';
  loading?: boolean;
}

interface AIConfig {
  symbols: string[];
  positionSize: number;
  stopLossPct: number;
  takeProfitPct: number;
  minConfidence: number;
  maxPositions: number;
  riskLimit: number;
}

interface AIStatus {
  totalSignals: number;
  successfulTrades: number;
  totalPnL: number;
  winRate: number;
  activePositions: number;
}

function AITradingControlsComponent({ 
  onStartAITrading, 
  onStopAITrading, 
  onUpdateAIConfig,
  aiTradingStatus,
  loading = false 
}: AITradingControlsProps) {
  const [showSettings, setShowSettings] = useState(false);
  const [aiStatus, setAIStatus] = useState<AIStatus>({
    totalSignals: 0,
    successfulTrades: 0,
    totalPnL: 0,
    winRate: 0,
    activePositions: 0
  });
  
  const [config, setConfig] = useState<AIConfig>({
    symbols: ['BTCUSDC', 'ETHUSDC'],
    positionSize: 1000,
    stopLossPct: 2.0,
    takeProfitPct: 4.0,
    minConfidence: 0.7,
    maxPositions: 5,
    riskLimit: 10000
  });

  // Fetch AI status periodically when active
  useEffect(() => {
    if (aiTradingStatus === 'active') {
      const interval = setInterval(async () => {
        try {
          const response = await fetch('http://127.0.0.1:8000/api/v1/ai-trading/status');
          if (response.ok) {
            const data = await response.json();
            setAIStatus({
              totalSignals: data.total_signals || 0,
              successfulTrades: data.successful_trades || 0,
              totalPnL: data.total_pnl || 0,
              winRate: data.win_rate || 0,
              activePositions: data.active_positions || 0
            });
          }
        } catch (error) {
          console.error('Error fetching AI status:', error);
        }
      }, 2000);
      
      return () => clearInterval(interval);
    }
  }, [aiTradingStatus]);

  const handleStartAI = async () => {
    await onStartAITrading(config.symbols);
  };

  const handleUpdateConfig = async () => {
    await onUpdateAIConfig({
      position_size: config.positionSize,
      stop_loss_pct: config.stopLossPct,
      take_profit_pct: config.takeProfitPct,
      min_confidence: config.minConfidence,
      max_positions: config.maxPositions,
      risk_limit: config.riskLimit
    });
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Brain className="w-6 h-6 text-purple-400" />
          <h2 className="text-xl font-semibold text-purple-400">AI Auto Trading</h2>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${
            aiTradingStatus === 'active' ? 'bg-green-500 animate-pulse' : 
            aiTradingStatus === 'inactive' ? 'bg-red-500' : 'bg-gray-500'
          }`}></div>
          <span className="text-sm text-gray-400">
            {aiTradingStatus === 'active' ? 'AI Trading Active' : 
             aiTradingStatus === 'inactive' ? 'AI Trading Stopped' : 'Unknown'}
          </span>
        </div>
      </div>

      {/* AI Performance Metrics */}
      {aiTradingStatus === 'active' && (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-4 p-3 bg-gray-900 rounded">
          <div className="text-center">
            <div className="text-lg font-bold text-blue-400">{aiStatus.totalSignals}</div>
            <div className="text-xs text-gray-400">Signals</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-green-400">{aiStatus.successfulTrades}</div>
            <div className="text-xs text-gray-400">Trades</div>
          </div>
          <div className="text-center">
            <div className={`text-lg font-bold ${aiStatus.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${aiStatus.totalPnL.toFixed(2)}
            </div>
            <div className="text-xs text-gray-400">P&L</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-yellow-400">{(aiStatus.winRate * 100).toFixed(1)}%</div>
            <div className="text-xs text-gray-400">Win Rate</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-orange-400">{aiStatus.activePositions}</div>
            <div className="text-xs text-gray-400">Positions</div>
          </div>
        </div>
      )}

      {/* Control Buttons */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <button
          onClick={handleStartAI}
          disabled={loading || aiTradingStatus === 'active'}
          className="flex items-center justify-center space-x-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white px-4 py-3 rounded transition-colors"
        >
          <Zap className="w-4 h-4" />
          <span>Start AI Trading</span>
        </button>

        <button
          onClick={onStopAITrading}
          disabled={loading || aiTradingStatus === 'inactive'}
          className="flex items-center justify-center space-x-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white px-4 py-3 rounded transition-colors"
        >
          <Pause className="w-4 h-4" />
          <span>Stop AI Trading</span>
        </button>
      </div>

      {/* Settings Toggle */}
      <button
        onClick={() => setShowSettings(!showSettings)}
        className="w-full flex items-center justify-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded transition-colors mb-4"
      >
        <Settings className="w-4 h-4" />
        <span>AI Configuration</span>
      </button>

      {/* AI Configuration Panel */}
      {showSettings && (
        <div className="border-t border-gray-700 pt-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Trading Symbols</label>
              <input
                type="text"
                value={config.symbols.join(', ')}
                onChange={(e) => setConfig(prev => ({ 
                  ...prev, 
                  symbols: e.target.value.split(',').map(s => s.trim()).filter(s => s) 
                }))}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                placeholder="BTCUSDC, ETHUSDC"
                aria-label="Trading symbols"
              />
            </div>
            
            <div>
              <label className="block text-sm text-gray-400 mb-1">Position Size ($)</label>
              <input
                type="number"
                value={config.positionSize}
                onChange={(e) => setConfig(prev => ({ ...prev, positionSize: Number(e.target.value) }))}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                min="100"
                max="50000"
                step="100"
                aria-label="Position size"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Stop Loss (%)</label>
              <input
                type="number"
                value={config.stopLossPct}
                onChange={(e) => setConfig(prev => ({ ...prev, stopLossPct: Number(e.target.value) }))}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                min="0.5"
                max="10"
                step="0.1"
                aria-label="Stop loss percent"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Take Profit (%)</label>
              <input
                type="number"
                value={config.takeProfitPct}
                onChange={(e) => setConfig(prev => ({ ...prev, takeProfitPct: Number(e.target.value) }))}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                min="1"
                max="20"
                step="0.1"
                aria-label="Take profit percent"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Min Confidence</label>
              <input
                type="number"
                value={config.minConfidence}
                onChange={(e) => setConfig(prev => ({ ...prev, minConfidence: Number(e.target.value) }))}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                min="0.1"
                max="1.0"
                step="0.05"
                aria-label="Minimum confidence"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Max Positions</label>
              <input
                type="number"
                value={config.maxPositions}
                onChange={(e) => setConfig(prev => ({ ...prev, maxPositions: Number(e.target.value) }))}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                min="1"
                max="20"
                aria-label="Maximum positions"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Risk Limit ($)</label>
            <input
              type="number"
              value={config.riskLimit}
              onChange={(e) => setConfig(prev => ({ ...prev, riskLimit: Number(e.target.value) }))}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
              min="1000"
              max="100000"
              step="1000"
              aria-label="Risk limit"
            />
          </div>

          <button
            onClick={handleUpdateConfig}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors"
          >
            Update AI Configuration
          </button>

          <div className="flex items-center space-x-2 text-sm text-yellow-400 bg-yellow-400/10 p-3 rounded">
            <AlertTriangle className="w-4 h-4" />
            <span>AI will automatically execute trades based on model predictions</span>
          </div>
        </div>
      )}
    </div>
  );
}

const AITradingControls = memo(AITradingControlsComponent, (prev, next) => {
  return prev.aiTradingStatus === next.aiTradingStatus && prev.loading === next.loading;
});

export default AITradingControls;