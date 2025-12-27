import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface AIData {
  accuracy: number;
  sharpe: number;
  latency: number;
  models: string[];
}

interface Prediction {
  id: string;
  timestamp: string;
  symbol: string;
  side: string;
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  leverage: number;
  model: string;
  reason: string;
  volatility: number;
  regime: string;
  position_size_usd: number;
}

interface PredictionsData {
  predictions: Prediction[];
  count: number;
  timestamp: number;
}

export default function AIEngine() {
  const [data, setData] = useState<AIData | null>(null);
  const [predictions, setPredictions] = useState<PredictionsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAI = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/ai/status`);
        if (!response.ok) throw new Error('Failed to fetch');
        const aiData = await response.json();
        setData(aiData);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load AI data:', err);
        setLoading(false);
      }
    };

    const fetchPredictions = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/ai/predictions`);
        if (!response.ok) throw new Error('Failed to fetch predictions');
        const predData = await response.json();
        setPredictions(predData);
      } catch (err) {
        console.error('Failed to load predictions:', err);
      }
    };

    fetchAI();
    fetchPredictions();
    const interval = setInterval(() => {
      fetchAI();
      fetchPredictions();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading AI Engine status...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400">Failed to load AI data</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-blue-400">AI Engine Status</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <InsightCard
          title="Model Accuracy"
          value={`${(data.accuracy * 100).toFixed(1)}%`}
          subtitle="Prediction accuracy over 24h"
          color="text-blue-400"
        />
        
        <InsightCard
          title="Sharpe Ratio"
          value={data.sharpe.toFixed(2)}
          subtitle="Risk-adjusted returns"
          color="text-green-400"
        />
        
        <InsightCard
          title="Latency"
          value={`${data.latency}ms`}
          subtitle="Average prediction time"
          color="text-purple-400"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Model Performance</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Accuracy</span>
                <span className="text-white">{(data.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-blue-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${data.accuracy * 100}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Sharpe Ratio</span>
                <span className="text-white">{data.sharpe.toFixed(2)}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-green-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(data.sharpe * 20, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Performance Metrics</h2>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Active Models</span>
              <span className="text-2xl font-bold text-white">{data.models?.length || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Avg. Latency</span>
              <span className="text-2xl font-bold text-white">{data.latency}ms</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Ensemble Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {(data.models || []).map((model) => (
            <div key={model} className="bg-gray-700 rounded p-4 text-center">
              <div className="text-sm text-gray-400">{model}</div>
              <div className="text-xl font-bold text-green-400 mt-1">Active</div>
            </div>
          ))}
        </div>
      </div>

      {/* AI Predictions Section */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">
          Live AI Predictions 
          <span className="text-sm text-gray-400 ml-2">
            ({predictions?.count || 0} recent signals)
          </span>
        </h2>
        
        {predictions && predictions.predictions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-400 border-b border-gray-700">
                  <th className="pb-3">Time</th>
                  <th className="pb-3">Symbol</th>
                  <th className="pb-3">Side</th>
                  <th className="pb-3">Confidence</th>
                  <th className="pb-3">Entry</th>
                  <th className="pb-3">TP/SL</th>
                  <th className="pb-3">Leverage</th>
                  <th className="pb-3">Model</th>
                  <th className="pb-3">Reason</th>
                </tr>
              </thead>
              <tbody>
                {predictions.predictions.slice(0, 5).map((pred) => (
                  <tr key={pred.id} className="border-b border-gray-700 text-sm">
                    <td className="py-3 text-gray-300">
                      {new Date(pred.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-3">
                      <span className="font-semibold text-white">{pred.symbol}</span>
                    </td>
                    <td className="py-3">
                      <span className={`font-bold ${pred.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                        {pred.side}
                      </span>
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-gray-700 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${pred.confidence > 0.7 ? 'bg-green-500' : pred.confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                            style={{ width: `${pred.confidence * 100}%` }}
                          />
                        </div>
                        <span className={`text-xs font-semibold ${pred.confidence > 0.7 ? 'text-green-400' : pred.confidence > 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 text-white font-mono text-xs">
                      ${pred.entry_price.toFixed(4)}
                    </td>
                    <td className="py-3">
                      <div className="text-xs">
                        <div className="text-green-400">TP: ${pred.take_profit.toFixed(4)}</div>
                        <div className="text-red-400">SL: ${pred.stop_loss.toFixed(4)}</div>
                      </div>
                    </td>
                    <td className="py-3">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${pred.leverage > 10 ? 'bg-red-900 text-red-200' : 'bg-blue-900 text-blue-200'}`}>
                        {pred.leverage}x
                      </span>
                    </td>
                    <td className="py-3 text-xs text-gray-400">
                      {pred.model}
                    </td>
                    <td className="py-3 text-xs text-gray-400 max-w-xs truncate">
                      {pred.reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            No predictions available
          </div>
        )}
      </div>
    </div>
  );
}
