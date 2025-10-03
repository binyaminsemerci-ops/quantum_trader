import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Globe, Brain, BarChart3, Radio } from 'lucide-react';

interface EnhancedData {
  sources: number;
  coingecko: { status: string; last_update: string };
  fear_greed: { value: number; classification: string };
  reddit_sentiment: { btc: number; eth: number; ada: number };
  cryptocompare_news: { count: number; sentiment: string };
  coinpaprika: { market_data: string };
  messari: { onchain_data: string };
  ai_insights: {
    market_regime: string;
    volatility: number;
    trend_strength: number;
    sentiment_score: number;
  };
}

const EnhancedDataDashboard: React.FC = () => {
  const [enhancedData, setEnhancedData] = useState<EnhancedData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    const fetchEnhancedData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/enhanced/data');
        if (response.ok) {
          const data = await response.json();
          setEnhancedData(data);
          setIsConnected(true);
          setLastUpdate(new Date());
        } else {
          throw new Error('Failed to fetch enhanced data');
        }
      } catch (error) {
        console.error('Error fetching enhanced data:', error);
        setIsConnected(false);
      }
    };

    // Initial fetch
    fetchEnhancedData();

    // Poll every 30 seconds
    const interval = setInterval(fetchEnhancedData, 30000);

    return () => clearInterval(interval);
  }, []);

  if (!enhancedData) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">📡 Enhanced Data Feeds</h3>
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-white">📡 Enhanced Data Feeds</h3>
        <div className={`px-2 py-1 rounded text-sm ${
          isConnected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
        }`}>
          {isConnected ? `${enhancedData.sources} Sources Active` : 'Disconnected'}
        </div>
      </div>

      <div className="space-y-4">
        {/* Market Sentiment Overview */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Brain className="text-blue-400" size={20} />
            <h4 className="text-white font-medium">AI Market Insights</h4>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-xs text-gray-400">Market Regime</div>
              <div className={`text-sm font-medium ${
                enhancedData.ai_insights.market_regime === 'BULL' ? 'text-green-400' : 'text-red-400'
              }`}>
                {enhancedData.ai_insights.market_regime}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Volatility</div>
              <div className="text-sm font-medium text-white">
                {enhancedData.ai_insights.volatility.toFixed(1)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Trend Strength</div>
              <div className="text-sm font-medium text-white">
                {enhancedData.ai_insights.trend_strength.toFixed(1)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Sentiment Score</div>
              <div className={`text-sm font-medium ${
                enhancedData.ai_insights.sentiment_score >= 0.5 ? 'text-green-400' : 'text-red-400'
              }`}>
                {(enhancedData.ai_insights.sentiment_score * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>

        {/* Fear & Greed Index */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <BarChart3 className="text-yellow-400" size={20} />
            <h4 className="text-white font-medium">Fear & Greed Index</h4>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-2xl font-bold text-white">{enhancedData.fear_greed.value}</span>
            <span className={`px-2 py-1 rounded text-sm ${
              enhancedData.fear_greed.classification === 'Neutral' ? 'bg-yellow-900 text-yellow-300' :
              enhancedData.fear_greed.classification === 'Greed' ? 'bg-green-900 text-green-300' :
              'bg-red-900 text-red-300'
            }`}>
              {enhancedData.fear_greed.classification}
            </span>
          </div>
        </div>

        {/* Social Sentiment */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Globe className="text-purple-400" size={20} />
            <h4 className="text-white font-medium">Social Sentiment</h4>
          </div>
          <div className="space-y-2">
            {Object.entries(enhancedData.reddit_sentiment).map(([symbol, sentiment]) => (
              <div key={symbol} className="flex justify-between items-center">
                <span className="text-gray-400 uppercase">{symbol}</span>
                <div className="flex items-center space-x-2">
                  {sentiment > 0 ? (
                    <TrendingUp className="text-green-400" size={16} />
                  ) : (
                    <TrendingDown className="text-red-400" size={16} />
                  )}
                  <span className={`text-sm ${
                    sentiment > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {(sentiment * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Data Sources Status */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Radio className="text-green-400" size={20} />
            <h4 className="text-white font-medium">Data Sources</h4>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">CoinGecko</span>
              <span className="text-green-400">●</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">CoinPaprika</span>
              <span className="text-green-400">●</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Messari</span>
              <span className="text-green-400">●</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">News API</span>
              <span className="text-green-400">●</span>
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-400">
            News Count: {enhancedData.cryptocompare_news.count} • Sentiment: {enhancedData.cryptocompare_news.sentiment}
          </div>
        </div>

        {lastUpdate && (
          <div className="text-xs text-gray-400 text-center">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedDataDashboard;