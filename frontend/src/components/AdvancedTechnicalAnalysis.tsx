import { useState, useEffect } from 'react';

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  strength: number; // 1-100
}

interface MarketSentiment {
  fearGreedIndex: number;
  socialSentiment: number;
  volumeProfile: number;
  institutionalFlow: number;
}

export default function AdvancedTechnicalAnalysis(): JSX.Element {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  
  const [technicalIndicators, setTechnicalIndicators] = useState<TechnicalIndicator[]>([
    { name: 'RSI (14)', value: 58.4, signal: 'NEUTRAL', strength: 60 },
    { name: 'MACD', value: 125.6, signal: 'BUY', strength: 75 },
    { name: 'Bollinger Bands', value: 0.85, signal: 'BUY', strength: 70 },
    { name: 'Stochastic', value: 42.1, signal: 'NEUTRAL', strength: 45 },
    { name: 'Williams %R', value: -35.8, signal: 'BUY', strength: 65 },
    { name: 'ADX', value: 32.5, signal: 'BUY', strength: 80 }
  ]);

  const [marketSentiment, setMarketSentiment] = useState<MarketSentiment>({
    fearGreedIndex: 68, // 0-100 (0=Extreme Fear, 100=Extreme Greed)
    socialSentiment: 72,
    volumeProfile: 85,
    institutionalFlow: 63
  });

  const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'];
  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

  // Simulate real-time indicator updates
  useEffect(() => {
    const interval = setInterval(() => {
      setTechnicalIndicators(prev => prev.map(indicator => ({
        ...indicator,
        value: indicator.value + (Math.random() - 0.5) * 2,
        strength: Math.max(10, Math.min(90, indicator.strength + (Math.random() - 0.5) * 10))
      })));

      setMarketSentiment(prev => ({
        ...prev,
        fearGreedIndex: Math.max(0, Math.min(100, prev.fearGreedIndex + (Math.random() - 0.5) * 5)),
        socialSentiment: Math.max(0, Math.min(100, prev.socialSentiment + (Math.random() - 0.5) * 8)),
        volumeProfile: Math.max(0, Math.min(100, prev.volumeProfile + (Math.random() - 0.5) * 6))
      }));
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-800';
      case 'SELL': return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-800';
      default: return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-800';
    }
  };

  const getStrengthColor = (strength: number) => {
    if (strength >= 70) return 'bg-green-500';
    if (strength >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getSentimentLevel = (value: number) => {
    if (value >= 75) return 'Extreme Bullish';
    if (value >= 60) return 'Bullish';
    if (value >= 40) return 'Neutral';
    if (value >= 25) return 'Bearish';
    return 'Extreme Bearish';
  };

  const getSentimentColor = (value: number) => {
    if (value >= 75) return 'text-green-700 bg-green-100';
    if (value >= 60) return 'text-green-600 bg-green-50';
    if (value >= 40) return 'text-yellow-600 bg-yellow-50';
    if (value >= 25) return 'text-orange-600 bg-orange-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
          🔬 Technical Analysis Hub
        </h3>
        <div className="flex items-center space-x-3">
          <select 
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            title="Select trading symbol"
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500"
          >
            {symbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
          <select 
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            title="Select timeframe"
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-white focus:ring-2 focus:ring-blue-500"
          >
            {timeframes.map(tf => (
              <option key={tf} value={tf}>{tf}</option>
            ))}
          </select>
          <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Technical Indicators */}
        <div>
          <h4 className="text-md font-semibold mb-4 text-gray-700 dark:text-gray-300">📊 Technical Indicators</h4>
          <div className="space-y-3">
            {technicalIndicators.map((indicator, index) => (
              <div key={index} className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg border dark:border-gray-600">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-800 dark:text-white">{indicator.name}</span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getSignalColor(indicator.signal)}`}>
                    {indicator.signal}
                  </span>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-lg font-bold text-gray-800 dark:text-white">
                    {indicator.value.toFixed(2)}
                  </span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Strength: {indicator.strength.toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2 relative">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${getStrengthColor(indicator.strength)}`}
                    data-width={indicator.strength}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Market Sentiment */}
        <div>
          <h4 className="text-md font-semibold mb-4 text-gray-700 dark:text-gray-300">🌡️ Market Sentiment</h4>
          
          {/* Fear & Greed Index */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg border dark:border-gray-600 mb-4">
            <div className="flex items-center justify-between mb-3">
              <span className="font-medium text-gray-800 dark:text-white">Fear & Greed Index</span>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(marketSentiment.fearGreedIndex)}`}>
                {getSentimentLevel(marketSentiment.fearGreedIndex)}
              </span>
            </div>
            <div className="text-3xl font-bold text-center mb-2 text-gray-800 dark:text-white">
              {marketSentiment.fearGreedIndex}
            </div>
            <div className="w-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full h-3 relative">
              <div 
                className="absolute top-0 w-3 h-3 bg-white border-2 border-gray-800 rounded-full transform -translate-x-1.5 left-1/2"
                data-position={marketSentiment.fearGreedIndex}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-2">
              <span>Extreme Fear</span>
              <span>Neutral</span>
              <span>Extreme Greed</span>
            </div>
          </div>

          {/* Other Sentiment Metrics */}
          <div className="space-y-3">
            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border dark:border-gray-600">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Social Sentiment</span>
                <span className="text-sm font-bold text-gray-800 dark:text-white">{marketSentiment.socialSentiment}%</span>
              </div>
              <div className="w-full bg-gray-300 dark:bg-gray-600 rounded-full h-2 mt-2 relative">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500 w-3/4"
                  data-width={marketSentiment.socialSentiment}
                ></div>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border dark:border-gray-600">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Volume Profile</span>
                <span className="text-sm font-bold text-gray-800 dark:text-white">{marketSentiment.volumeProfile}%</span>
              </div>
              <div className="w-full bg-gray-300 dark:bg-gray-600 rounded-full h-2 mt-2 relative">
                <div 
                  className="bg-purple-500 h-2 rounded-full transition-all duration-500 w-5/6"
                  data-width={marketSentiment.volumeProfile}
                ></div>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg border dark:border-gray-600">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Institutional Flow</span>
                <span className="text-sm font-bold text-gray-800 dark:text-white">{marketSentiment.institutionalFlow}%</span>
              </div>
              <div className="w-full bg-gray-300 dark:bg-gray-600 rounded-full h-2 mt-2 relative">
                <div 
                  className="bg-orange-500 h-2 rounded-full transition-all duration-500 w-2/3"
                  data-width={marketSentiment.institutionalFlow}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}