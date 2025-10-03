/**
 * Enhanced Data Dashboard - Multi-Source API Integration Display
 * 
 * Shows real-time data from:
 * - CoinGecko market data & global metrics
 * - Fear & Greed Index from Alternative.me
 * - Reddit sentiment analysis
 * - CryptoCompare news feeds
 * - CoinPaprika additional metrics
 * - Messari on-chain data
 * - AI-extracted insights
 */

import React, { useState, useEffect } from 'react';

interface EnhancedData {
  coingecko: {
    market_data: Array<{
      id: string;
      symbol: string;
      name: string;
      current_price: number;
      price_change_percentage_24h: number;
      market_cap_rank: number;
      total_volume: number;
      market_cap: number;
    }>;
    global_data: {
      total_market_cap: { usd: number };
      market_cap_percentage: { btc: number };
      active_cryptocurrencies: number;
    };
  };
  fear_greed: {
    current: {
      value: number;
      value_classification: string;
      timestamp: string;
    };
  };
  reddit: {
    symbols: Record<string, {
      sentiment_score: number;
      total_posts: number;
      total_engagement: number;
    }>;
  };
  news: {
    news: Array<{
      id: string;
      title: string;
      published_at: string;
      url?: string;
    }>;
  };
  ai_insights: {
    market_sentiment: string;
    momentum_signals: string[];
    risk_factors: string[];
    opportunity_signals: string[];
    regime_indicators: {
      btc_dominance: number;
      total_market_cap: number;
      market_stage: string;
    };
  };
  timestamp: string;
}

interface Props {
  wsUrl: string;
}

const EnhancedDataDashboard: React.FC<Props> = ({ wsUrl }) => {
  const [enhancedData, setEnhancedData] = useState<EnhancedData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('🔗 Enhanced Data WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'enhanced_data_update') {
          setEnhancedData(data.data);
          setLastUpdate(new Date());
        }
      } catch (error) {
        console.error('Error parsing enhanced data:', error);
      }
    };

    ws.onclose = () => {
      console.log('🔌 Enhanced Data WebSocket disconnected');
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('Enhanced Data WebSocket error:', error);
    };

    return () => ws.close();
  }, [wsUrl]);

  const formatCurrency = (value: number): string => {
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
    return `$${value.toFixed(2)}`;
  };

  const getFearGreedColor = (value: number): string => {
    if (value <= 25) return 'text-red-500';
    if (value <= 45) return 'text-orange-500';
    if (value <= 55) return 'text-yellow-500';
    if (value <= 75) return 'text-green-500';
    return 'text-red-400'; // Extreme greed
  };

  const getSentimentColor = (sentiment: string): string => {
    switch (sentiment.toLowerCase()) {
      case 'bullish': case 'extreme_greed': return 'text-green-500';
      case 'bearish': case 'extreme_fear': return 'text-red-500';
      case 'neutral': return 'text-gray-400';
      default: return 'text-blue-400';
    }
  };

  if (!enhancedData) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-bold text-white mb-4">📡 Enhanced Data Sources</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-gray-300">
            {isConnected ? 'Connected, waiting for data...' : 'Connecting to enhanced data feed...'}
          </span>
        </div>
      </div>
    );
  }

  const { coingecko, fear_greed, reddit, news, ai_insights } = enhancedData;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-bold text-white">📡 Enhanced Multi-Source Data</h3>
          <div className="flex items-center space-x-4">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm text-gray-400">
              Last update: {lastUpdate?.toLocaleTimeString() || 'Never'}
            </span>
          </div>
        </div>
      </div>

      {/* AI Insights */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h4 className="text-lg font-semibold text-white mb-3">🧠 AI Market Insights</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-300">Market Sentiment:</span>
              <span className={`font-medium ${getSentimentColor(ai_insights.market_sentiment)}`}>
                {ai_insights.market_sentiment.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Active Signals:</span>
              <span className="text-green-400 font-medium">
                {ai_insights.momentum_signals.length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Risk Factors:</span>
              <span className="text-red-400 font-medium">
                {ai_insights.risk_factors.length}
              </span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-300">BTC Dominance:</span>
              <span className="text-blue-400 font-medium">
                {ai_insights.regime_indicators.btc_dominance.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Market Stage:</span>
              <span className="text-purple-400 font-medium">
                {ai_insights.regime_indicators.market_stage.replace('_', ' ').toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Opportunities:</span>
              <span className="text-yellow-400 font-medium">
                {ai_insights.opportunity_signals.length}
              </span>
            </div>
          </div>
        </div>

        {/* Signals Detail */}
        {ai_insights.momentum_signals.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-700">
            <h5 className="text-sm font-medium text-gray-300 mb-2">Active Momentum Signals:</h5>
            <div className="flex flex-wrap gap-2">
              {ai_insights.momentum_signals.slice(0, 5).map((signal, index) => (
                <span key={index} className="px-2 py-1 bg-green-900 text-green-300 rounded text-xs">
                  {signal.replace('_', ' ')}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Fear & Greed Index + Global Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-3">😱 Fear & Greed Index</h4>
          <div className="text-center">
            <div className={`text-4xl font-bold ${getFearGreedColor(fear_greed.current.value)}`}>
              {fear_greed.current.value}
            </div>
            <div className={`text-sm ${getFearGreedColor(fear_greed.current.value)}`}>
              {fear_greed.current.value_classification}
            </div>
            <div className="text-xs text-gray-400 mt-2">
              Alternative.me Index
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-3">🌍 Global Market</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-300">Total Market Cap:</span>
              <span className="text-blue-400 font-medium">
                {formatCurrency(coingecko.global_data.total_market_cap.usd)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">BTC Dominance:</span>
              <span className="text-orange-400 font-medium">
                {coingecko.global_data.market_cap_percentage.btc.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Active Cryptos:</span>
              <span className="text-green-400 font-medium">
                {coingecko.global_data.active_cryptocurrencies.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Top Market Data */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h4 className="text-lg font-semibold text-white mb-3">📈 Top Market Movers (CoinGecko)</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left text-gray-300 py-2">Rank</th>
                <th className="text-left text-gray-300 py-2">Coin</th>
                <th className="text-right text-gray-300 py-2">Price</th>
                <th className="text-right text-gray-300 py-2">24h Change</th>
                <th className="text-right text-gray-300 py-2">Market Cap</th>
              </tr>
            </thead>
            <tbody>
              {coingecko.market_data.slice(0, 5).map((coin) => (
                <tr key={coin.id} className="border-b border-gray-700/50">
                  <td className="py-2 text-gray-400">#{coin.market_cap_rank}</td>
                  <td className="py-2">
                    <div>
                      <div className="font-medium text-white">{coin.name}</div>
                      <div className="text-xs text-gray-400">{coin.symbol.toUpperCase()}</div>
                    </div>
                  </td>
                  <td className="py-2 text-right text-white font-medium">
                    ${coin.current_price.toLocaleString()}
                  </td>
                  <td className={`py-2 text-right font-medium ${
                    coin.price_change_percentage_24h > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {coin.price_change_percentage_24h > 0 ? '+' : ''}{coin.price_change_percentage_24h.toFixed(2)}%
                  </td>
                  <td className="py-2 text-right text-gray-300">
                    {formatCurrency(coin.market_cap)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Reddit Sentiment */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h4 className="text-lg font-semibold text-white mb-3">🔴 Reddit Sentiment Analysis</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {Object.entries(reddit.symbols).map(([symbol, data]) => (
            <div key={symbol} className="bg-gray-700 rounded p-3">
              <div className="font-medium text-white">{symbol}</div>
              <div className={`text-sm ${
                data.sentiment_score > 0.1 ? 'text-green-400' : 
                data.sentiment_score < -0.1 ? 'text-red-400' : 'text-gray-400'
              }`}>
                Sentiment: {data.sentiment_score.toFixed(2)}
              </div>
              <div className="text-xs text-gray-400">
                {data.total_posts} posts, {data.total_engagement} engagement
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent News */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h4 className="text-lg font-semibold text-white mb-3">📰 Latest Crypto News</h4>
        <div className="space-y-3">
          {news.news.slice(0, 5).map((item) => (
            <div key={item.id} className="border-b border-gray-700 pb-3">
              <div className="text-white font-medium hover:text-blue-400 cursor-pointer">
                {item.title}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {new Date(item.published_at).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default EnhancedDataDashboard;