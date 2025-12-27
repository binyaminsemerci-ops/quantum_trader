import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface AIData {
  model_accuracy: number;
  sharpe_ratio: number;
  prediction_latency_ms: number;
  daily_signals: number;
  ensemble_confidence: number;
}

export default function AIEngine() {
  const [data, setData] = useState<AIData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAI = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/ai/status`);
        const aiData = await response.json();
        setData(aiData);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load AI data:', err);
        setLoading(false);
      }
    };

    fetchAI();
    const interval = setInterval(fetchAI, 5000);
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
          value={`${(data.model_accuracy * 100).toFixed(2)}%`}
          subtitle="Prediction accuracy over 24h"
          color="text-blue-400"
        />
        
        <InsightCard
          title="Sharpe Ratio"
          value={data.sharpe_ratio.toFixed(3)}
          subtitle="Risk-adjusted returns"
          color="text-green-400"
        />
        
        <InsightCard
          title="Latency"
          value={`${data.prediction_latency_ms}ms`}
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
                <span className="text-white">{(data.model_accuracy * 100).toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-blue-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${data.model_accuracy * 100}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Confidence</span>
                <span className="text-white">{(data.ensemble_confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-green-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${data.ensemble_confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Daily Activity</h2>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Signals Generated</span>
              <span className="text-2xl font-bold text-white">{data.daily_signals}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Avg. Latency</span>
              <span className="text-2xl font-bold text-white">{data.prediction_latency_ms}ms</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Ensemble Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {['XGBoost', 'LightGBM', 'N-HiTS', 'TFT'].map((model) => (
            <div key={model} className="bg-gray-700 rounded p-4 text-center">
              <div className="text-sm text-gray-400">{model}</div>
              <div className="text-xl font-bold text-green-400 mt-1">Active</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
