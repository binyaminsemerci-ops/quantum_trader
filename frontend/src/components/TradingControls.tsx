import { useState, memo } from 'react';
import { Play, Pause, RotateCcw, Settings } from 'lucide-react';

interface TradingControlsProps {
  onStartTrading: () => Promise<void>;
  onStopTrading: () => Promise<void>;
  onRunCycle: () => Promise<void>;
  tradingStatus: 'active' | 'inactive' | 'unknown';
  loading?: boolean;
}

function TradingControlsComponent({ 
  onStartTrading, 
  onStopTrading, 
  onRunCycle,
  tradingStatus,
  loading = false 
}: TradingControlsProps) {
  const [showSettings, setShowSettings] = useState(false);
  const [config, setConfig] = useState({
    intervalMinutes: 5,
    maxRisk: 0.02,
    stopLoss: 0.05
  });

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-orange-400">Trading Controls</h2>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${
            tradingStatus === 'active' ? 'bg-green-500' : 
            tradingStatus === 'inactive' ? 'bg-red-500' : 'bg-gray-500'
          }`}></div>
          <span className="text-sm text-gray-400">
            {tradingStatus === 'active' ? 'Trading Active' : 
             tradingStatus === 'inactive' ? 'Trading Stopped' : 'Unknown'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <button
          onClick={onStartTrading}
          disabled={loading || tradingStatus === 'active'}
          className="flex items-center justify-center space-x-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white px-4 py-2 rounded transition-colors"
        >
          <Play className="w-4 h-4" />
          <span>Start Trading</span>
        </button>

        <button
          onClick={onStopTrading}
          disabled={loading || tradingStatus === 'inactive'}
          className="flex items-center justify-center space-x-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white px-4 py-2 rounded transition-colors"
        >
          <Pause className="w-4 h-4" />
          <span>Stop Trading</span>
        </button>

        <button
          onClick={onRunCycle}
          disabled={loading}
          className="flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Run Cycle</span>
        </button>

        <button
          onClick={() => setShowSettings(!showSettings)}
          className="flex items-center justify-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded transition-colors"
        >
          <Settings className="w-4 h-4" />
          <span>Settings</span>
        </button>
      </div>

      {showSettings && (
        <div className="border-t border-gray-700 pt-4 space-y-3">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Interval (minutes)</label>
            <input
              type="number"
              value={config.intervalMinutes}
              onChange={(e) => setConfig(prev => ({ ...prev, intervalMinutes: Number(e.target.value) }))}
              className="w-full bg-gray-700 text-white px-3 py-1 rounded text-sm"
              min="1"
              max="60"
              aria-label="Interval minutes"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-400 mb-1">Max Risk (%)</label>
            <input
              type="number"
              value={config.maxRisk * 100}
              onChange={(e) => setConfig(prev => ({ ...prev, maxRisk: Number(e.target.value) / 100 }))}
              className="w-full bg-gray-700 text-white px-3 py-1 rounded text-sm"
              min="0.1"
              max="10"
              step="0.1"
              aria-label="Max risk percent"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Stop Loss (%)</label>
            <input
              type="number"
              value={config.stopLoss * 100}
              onChange={(e) => setConfig(prev => ({ ...prev, stopLoss: Number(e.target.value) / 100 }))}
              className="w-full bg-gray-700 text-white px-3 py-1 rounded text-sm"
              min="0.5"
              max="20"
              step="0.1"
              aria-label="Stop loss percent"
            />
          </div>
        </div>
      )}
    </div>
  );
}

const TradingControls = memo(TradingControlsComponent, (prev, next) => {
  return prev.tradingStatus === next.tradingStatus && prev.loading === next.loading;
});

export default TradingControls;