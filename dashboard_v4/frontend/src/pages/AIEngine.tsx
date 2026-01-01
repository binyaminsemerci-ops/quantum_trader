import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface ModelHealth {
  name: string;
  weight: number;
  mape: number;
  avg_pnl: number;
  drift_count: number;
  retrain_count: number;
  samples: number;
  status: string;
}

interface AIEngineMetrics {
  models_loaded: number;
  signals_generated_total: number;
  ensemble_enabled: boolean;
  governance_active: boolean;
  cross_exchange_intelligence: boolean;
  intelligent_leverage_v2: boolean;
  rl_position_sizing: boolean;
  adaptive_leverage_enabled: boolean;
}

interface ConsensusData {
  symbol: string;
  consensus_confidence: number;
  model_votes: { [key: string]: number };
  ensemble_decision: 'BUY' | 'SELL' | 'HOLD';
  model_count: number;
}

interface AIHealthData {
  status: string;
  version: string;
  uptime_seconds: number;
  metrics: AIEngineMetrics;
  governance: {
    active_models: number;
    drift_threshold: number;
    retrain_interval: number;
    last_retrain: string;
    models: { [key: string]: any };
  };
  dependencies: {
    redis: { status: string; latency_ms: number };
    eventbus: { status: string };
  };
}

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
  const [healthData, setHealthData] = useState<AIHealthData | null>(null);
  const [predictions, setPredictions] = useState<PredictionsData | null>(null);
  const [loading, setLoading] = useState(true);

  // Calculate consensus from recent predictions
  const calculateConsensus = (predictions: Prediction[]): ConsensusData[] => {
    const symbolMap: { [key: string]: { confidences: number[]; sides: string[] } } = {};
    
    predictions.forEach(pred => {
      if (!symbolMap[pred.symbol]) {
        symbolMap[pred.symbol] = { confidences: [], sides: [] };
      }
      symbolMap[pred.symbol].confidences.push(pred.confidence);
      symbolMap[pred.symbol].sides.push(pred.side);
    });

    return Object.entries(symbolMap).map(([symbol, data]) => {
      const avgConfidence = data.confidences.reduce((a, b) => a + b, 0) / data.confidences.length;
      const buyCount = data.sides.filter(s => s === 'LONG' || s === 'BUY').length;
      const sellCount = data.sides.filter(s => s === 'SHORT' || s === 'SELL').length;
      
      let ensemble_decision: 'BUY' | 'SELL' | 'HOLD';
      if (buyCount > sellCount) {
        ensemble_decision = 'BUY';
      } else if (sellCount > buyCount) {
        ensemble_decision = 'SELL';
      } else {
        ensemble_decision = 'HOLD';
      }
      
      return {
        symbol,
        consensus_confidence: avgConfidence,
        model_votes: {
          'LONG': buyCount,
          'SHORT': sellCount
        },
        ensemble_decision,
        model_count: data.confidences.length
      };
    }).sort((a, b) => b.consensus_confidence - a.consensus_confidence);
  };

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

    const fetchHealth = async () => {
      try {
        // Fetch directly from AI Engine health endpoint
        const response = await fetch('http://46.224.116.254:8001/health');
        if (!response.ok) throw new Error('Failed to fetch health');
        const health = await response.json();
        setHealthData(health);
      } catch (err) {
        console.error('Failed to load AI health:', err);
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
    fetchHealth();
    fetchPredictions();
    const interval = setInterval(() => {
      fetchAI();
      fetchHealth();
      fetchPredictions();
    }, 5000); // Refresh every 5s for live updates
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

  const consensusData = predictions ? calculateConsensus(predictions.predictions) : [];
  const modelHealth: ModelHealth[] = healthData?.governance?.models ? 
    Object.entries(healthData.governance.models).map(([name, data]: [string, any]) => ({
      name,
      weight: data.weight || 0,
      mape: data.last_mape || 0,
      avg_pnl: data.avg_pnl || 0,
      drift_count: data.drift_count || 0,
      retrain_count: data.retrain_count || 0,
      samples: data.samples || 0,
      status: data.drift_count > 3 ? 'warning' : 'healthy'
    })) : [];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-blue-400">AI Engine Status</h1>
        <div className="text-sm text-gray-400">
          🔴 Live • {healthData?.status === 'OK' ? '✅ Healthy' : '⚠️ ' + healthData?.status}
        </div>
      </div>
      
      {/* Main Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <InsightCard
          title="Ensemble Accuracy"
          value={`${(data.accuracy * 100).toFixed(1)}%`}
          subtitle={`${data.models.length} models active`}
          color="text-blue-400"
        />
        
        <InsightCard
          title="Signals Generated"
          value={healthData?.metrics?.signals_generated_total?.toLocaleString() || '0'}
          subtitle="Total AI predictions"
          color="text-green-400"
        />
        
        <InsightCard
          title="Models Loaded"
          value={`${healthData?.metrics?.models_loaded || 0}`}
          subtitle={`Governance: ${healthData?.governance?.active_models || 0} active`}
          color="text-purple-400"
        />
        
        <InsightCard
          title="Avg Latency"
          value={`${data.latency}ms`}
          subtitle={`Redis: ${healthData?.dependencies?.redis?.latency_ms?.toFixed(1) || 'N/A'}ms`}
          color="text-cyan-400"
        />
      </div>

      {/* Consensus Confidence Section */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-2">
          🎯 Ensemble Consensus Confidence
        </h2>
        <p className="text-sm text-gray-400 mb-4">
          Aggregated confidence from multiple AI models - Shows agreement level and ensemble decision
        </p>
        
        {consensusData.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {consensusData.slice(0, 6).map((consensus) => (
              <div key={consensus.symbol} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <div className="text-lg font-bold text-white">{consensus.symbol}</div>
                    <div className="text-xs text-gray-400">{consensus.model_count} model votes</div>
                  </div>
                  <div className={`px-3 py-1 rounded text-sm font-bold ${
                    consensus.ensemble_decision === 'BUY' ? 'bg-green-900 text-green-200' :
                    consensus.ensemble_decision === 'SELL' ? 'bg-red-900 text-red-200' :
                    'bg-gray-600 text-gray-200'
                  }`}>
                    {consensus.ensemble_decision}
                  </div>
                </div>
                
                {/* Consensus Confidence Bar */}
                <div className="mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">Consensus Confidence</span>
                    <span className={`font-bold ${
                      consensus.consensus_confidence > 0.7 ? 'text-green-400' :
                      consensus.consensus_confidence > 0.5 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {(consensus.consensus_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-600 rounded-full h-3">
                    <div 
                      className={`h-3 rounded-full transition-all duration-300 ${
                        consensus.consensus_confidence > 0.7 ? 'bg-green-500' :
                        consensus.consensus_confidence > 0.5 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${consensus.consensus_confidence * 100}%` }}
                    />
                  </div>
                </div>
                
                {/* Model Votes */}
                <div className="flex gap-2 text-xs">
                  <div className="flex-1 bg-green-900/30 rounded px-2 py-1">
                    <span className="text-green-400">LONG: {consensus.model_votes.LONG || 0}</span>
                  </div>
                  <div className="flex-1 bg-red-900/30 rounded px-2 py-1">
                    <span className="text-red-400">SHORT: {consensus.model_votes.SHORT || 0}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            No consensus data available - Waiting for AI predictions
          </div>
        )}
      </div>

      {/* Ensemble Model Health */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-2">
          🏥 Ensemble Model Health & Performance
        </h2>
        <p className="text-sm text-gray-400 mb-4">
          Live status of each model in the ensemble - Weight, accuracy (MAPE), drift, and retraining history
        </p>
        
        {modelHealth.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {modelHealth.map((model) => (
              <div key={model.name} className="bg-gray-700 rounded-lg p-5 border-l-4" style={{
                borderLeftColor: model.status === 'healthy' ? '#10b981' : '#f59e0b'
              }}>
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-white">{model.name}</h3>
                    <div className="text-xs text-gray-400 mt-1">
                      {model.samples} samples • Weight: {(model.weight * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className={`px-3 py-1 rounded text-xs font-bold ${
                    model.status === 'healthy' ? 'bg-green-900 text-green-200' : 'bg-yellow-900 text-yellow-200'
                  }`}>
                    {model.status === 'healthy' ? '✅ Healthy' : '⚠️ Warning'}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-400">MAPE (Accuracy)</div>
                    <div className="text-lg font-bold text-green-400">
                      {(model.mape * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Avg PnL</div>
                    <div className={`text-lg font-bold ${model.avg_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ${model.avg_pnl.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Drift Events</div>
                    <div className={`text-lg font-bold ${model.drift_count > 2 ? 'text-yellow-400' : 'text-gray-300'}`}>
                      {model.drift_count}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Retrains</div>
                    <div className="text-lg font-bold text-blue-400">
                      {model.retrain_count}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            Model health data unavailable
          </div>
        )}
      </div>

      {/* System Features Status */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">🎛️ AI System Features</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { name: 'Ensemble', enabled: healthData?.metrics?.ensemble_enabled, icon: '🔗' },
            { name: 'Governance', enabled: healthData?.metrics?.governance_active, icon: '👑' },
            { name: 'Cross-Exchange', enabled: healthData?.metrics?.cross_exchange_intelligence, icon: '🌐' },
            { name: 'Intelligent Leverage', enabled: healthData?.metrics?.intelligent_leverage_v2, icon: '⚖️' },
            { name: 'RL Position Sizing', enabled: healthData?.metrics?.rl_position_sizing, icon: '🎯' },
            { name: 'Adaptive Leverage', enabled: healthData?.metrics?.adaptive_leverage_enabled, icon: '📊' },
          ].map((feature) => (
            <div key={feature.name} className={`p-4 rounded-lg border-2 ${
              feature.enabled ? 'border-green-500 bg-green-900/20' : 'border-gray-600 bg-gray-700'
            }`}>
              <div className="text-2xl mb-2">{feature.icon}</div>
              <div className="text-sm font-semibold text-white">{feature.name}</div>
              <div className={`text-xs mt-1 ${feature.enabled ? 'text-green-400' : 'text-gray-400'}`}>
                {feature.enabled ? '✅ Enabled' : '⭕ Disabled'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* AI Predictions Section */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">
          Live AI Predictions 
          <span className="text-sm text-gray-400 ml-2">
            (showing {Math.min(predictions?.predictions.length || 0, 10)} of {predictions?.count || 0} signals)
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
                {predictions.predictions.map((pred) => (
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
