import React, { useState, useEffect } from 'react';

interface ContinuousLearningStatus {
  is_running: boolean;
  symbols_monitored: number;
  last_training: number;
  model_performance: {
    accuracy: number;
    sharpe: number;
    total_trades: number;
  };
  current_regime: string;
  data_points_collected: number;
  adaptive_features: string[];
}

const ContinuousLearningDashboard: React.FC = () => {
  const [learningStatus, setLearningStatus] = useState<ContinuousLearningStatus | null>(null);
  const [twitterSentiment, setTwitterSentiment] = useState<any[]>([]);
  const [marketRegimes, setMarketRegimes] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchLearningStatus = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/ai-trading/status');
      const data = await response.json();
      
      if (data.continuous_learning) {
        setLearningStatus(data.continuous_learning);
      }
      
      setLastUpdate(new Date());
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch learning status:', error);
      setIsLoading(false);
    }
  };

  const fetchTwitterSentiment = async () => {
    try {
      // Mock Twitter sentiment data - replace with real API call
      const mockSentiment = [
        { symbol: 'BTC', sentiment: 0.65, label: 'positive', tweets: 245 },
        { symbol: 'ETH', sentiment: 0.32, label: 'neutral', tweets: 189 },
        { symbol: 'ADA', sentiment: -0.21, label: 'negative', tweets: 156 }
      ];
      setTwitterSentiment(mockSentiment);
    } catch (error) {
      console.error('Failed to fetch Twitter sentiment:', error);
    }
  };

  const fetchMarketRegimes = async () => {
    try {
      // Mock market regime data - replace with real API call
      const mockRegimes = [
        { time: '14:30', regime: 'bull', volatility: 0.025, trend: 0.032 },
        { time: '14:00', regime: 'volatile', volatility: 0.045, trend: 0.012 },
        { time: '13:30', regime: 'sideways', volatility: 0.018, trend: -0.008 }
      ];
      setMarketRegimes(mockRegimes);
    } catch (error) {
      console.error('Failed to fetch market regimes:', error);
    }
  };

  useEffect(() => {
    fetchLearningStatus();
    fetchTwitterSentiment();
    fetchMarketRegimes();

    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchLearningStatus();
      fetchTwitterSentiment();
      fetchMarketRegimes();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const getRegimeColor = (regime: string) => {
    switch (regime.toLowerCase()) {
      case 'bull': return 'bg-green-500';
      case 'bear': return 'bg-red-500';
      case 'volatile': return 'bg-orange-500';
      case 'sideways': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.2) return 'text-green-600';
    if (sentiment < -0.2) return 'text-red-600';
    return 'text-gray-600';
  };

  if (isLoading) {
    return (
      <div className="p-6 bg-white rounded-lg shadow-md">
        <div className="flex items-center space-x-2">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          <span>Loading continuous learning status...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <span className="text-purple-600">🧠</span>
            <span>Continuous Learning Engine</span>
          </h2>
          <p className="text-gray-600">Real-time AI strategy evolution from live data feeds</p>
        </div>
        <button 
          onClick={fetchLearningStatus} 
          className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
        >
          🔄 Refresh
        </button>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow-md">
          <div className="flex items-center space-x-2">
            <span className={`text-2xl ${learningStatus?.is_running ? '🟢' : '🔴'}`}>
              {learningStatus?.is_running ? '🟢' : '🔴'}
            </span>
            <div>
              <p className="text-sm font-medium">Learning Status</p>
              <p className="text-2xl font-bold">
                {learningStatus?.is_running ? 'Active' : 'Inactive'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">👁️</span>
            <div>
              <p className="text-sm font-medium">Symbols Monitored</p>
              <p className="text-2xl font-bold">{learningStatus?.symbols_monitored || 0}</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">💾</span>
            <div>
              <p className="text-sm font-medium">Data Points</p>
              <p className="text-2xl font-bold">{learningStatus?.data_points_collected || 0}</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">📈</span>
            <div>
              <p className="text-sm font-medium">Model Accuracy</p>
              <p className="text-2xl font-bold">
                {((learningStatus?.model_performance?.accuracy || 0) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Live Data Feeds */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Twitter Sentiment Feed */}
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-4 border-b">
            <h3 className="text-lg font-semibold flex items-center space-x-2">
              <span className="text-blue-400">🐦</span>
              <span>Live Twitter Sentiment</span>
              <span className="ml-auto px-2 py-1 bg-green-100 text-green-800 rounded-full text-sm">Live</span>
            </h3>
          </div>
          <div className="p-4">
            <div className="space-y-3">
              {twitterSentiment.map((item) => (
                <div key={item.symbol} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="font-medium">{item.symbol}</div>
                    <span className={`px-2 py-1 rounded-full text-sm ${getSentimentColor(item.sentiment)} bg-gray-100`}>
                      {item.label}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold ${getSentimentColor(item.sentiment)}`}>
                      {item.sentiment > 0 ? '+' : ''}{item.sentiment.toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-500">{item.tweets} tweets</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Market Regime Detection */}
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-4 border-b">
            <h3 className="text-lg font-semibold flex items-center space-x-2">
              <span className="text-yellow-500">⚡</span>
              <span>Market Regime Analysis</span>
              <span className="ml-auto px-2 py-1 bg-green-100 text-green-800 rounded-full text-sm">Live</span>
            </h3>
          </div>
          <div className="p-4">
            <div className="space-y-3">
              {marketRegimes.map((regime, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="text-sm font-medium text-gray-500">{regime.time}</div>
                    <span className={`px-2 py-1 rounded-full text-white text-sm ${getRegimeColor(regime.regime)}`}>
                      {regime.regime.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-right text-sm">
                    <div>Vol: {(regime.volatility * 100).toFixed(1)}%</div>
                    <div>Trend: {(regime.trend * 100).toFixed(1)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Learning Progress */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="p-4 border-b">
          <h3 className="text-lg font-semibold flex items-center space-x-2">
            <span className="text-purple-600">🧠</span>
            <span>AI Learning Progress</span>
          </h3>
        </div>
        <div className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            
            <div className="space-y-2">
              <p className="text-sm font-medium">Last Model Training</p>
              <p className="text-lg">
                {learningStatus?.last_training 
                  ? formatTimestamp(learningStatus.last_training)
                  : 'Never'
                }
              </p>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium">Current Market Regime</p>
              <span className={`px-3 py-1 rounded-full text-white text-sm ${getRegimeColor(learningStatus?.current_regime || 'unknown')}`}>
                {(learningStatus?.current_regime || 'Unknown').toUpperCase()}
              </span>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium">Adaptive Features</p>
              <div className="flex flex-wrap gap-1">
                {(learningStatus?.adaptive_features || []).slice(0, 3).map((feature) => (
                  <span key={feature} className="px-2 py-1 bg-gray-100 border rounded text-xs">
                    {feature}
                  </span>
                ))}
                {(learningStatus?.adaptive_features || []).length > 3 && (
                  <span className="px-2 py-1 bg-gray-100 border rounded text-xs">
                    +{(learningStatus?.adaptive_features || []).length - 3} more
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <div className="flex items-center space-x-2 text-blue-700">
              <span className="text-xl">📚</span>
              <span className="font-medium">Learning Insights</span>
            </div>
            <p className="text-sm text-blue-600 mt-2">
              🤖 AI is continuously analyzing Twitter sentiment, market data, and news feeds to evolve trading strategies. 
              The model retrains every hour with fresh data to adapt to changing market conditions.
            </p>
            <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="font-semibold text-blue-700">🐦 Twitter Feed</div>
                <div className="text-blue-600">Every 60 seconds</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-blue-700">📊 Market Data</div>
                <div className="text-blue-600">Every 30 seconds</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-blue-700">🤖 Model Training</div>
                <div className="text-blue-600">Every hour</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Last Updated */}
      <div className="text-center text-sm text-gray-500">
        Last updated: {lastUpdate.toLocaleTimeString()} • 
        <span className="text-green-600 ml-1">● Live Data Active</span>
      </div>
    </div>
  );
};

export default ContinuousLearningDashboard;