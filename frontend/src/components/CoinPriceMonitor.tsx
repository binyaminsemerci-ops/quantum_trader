import { useState, useEffect } from 'react';

interface CoinData {
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  volume24h: number;
  marketCap: number;
  logo: string;
}

const TOP_COINS: CoinData[] = [
  { symbol: 'BTC', name: 'Bitcoin', price: 67450.00, change24h: 2.45, volume24h: 28500000000, marketCap: 1320000000000, logo: '₿' },
  { symbol: 'ETH', name: 'Ethereum', price: 2680.50, change24h: -1.25, volume24h: 15200000000, marketCap: 322000000000, logo: 'Ξ' },
  { symbol: 'BNB', name: 'BNB', price: 584.20, change24h: 3.15, volume24h: 1800000000, marketCap: 87500000000, logo: '🟡' },
  { symbol: 'SOL', name: 'Solana', price: 142.80, change24h: 5.67, volume24h: 2100000000, marketCap: 67200000000, logo: '◎' },
  { symbol: 'ADA', name: 'Cardano', price: 0.3845, change24h: -2.10, volume24h: 450000000, marketCap: 13500000000, logo: '₳' },
  { symbol: 'MATIC', name: 'Polygon', price: 0.9234, change24h: 4.32, volume24h: 380000000, marketCap: 8900000000, logo: '⬟' },
  { symbol: 'DOT', name: 'Polkadot', price: 4.567, change24h: -0.85, volume24h: 180000000, marketCap: 6200000000, logo: '●' },
  { symbol: 'AVAX', name: 'Avalanche', price: 28.45, change24h: 6.78, volume24h: 520000000, marketCap: 11800000000, logo: '🔺' },
  { symbol: 'LINK', name: 'Chainlink', price: 11.23, change24h: 2.89, volume24h: 290000000, marketCap: 6800000000, logo: '🔗' },
  { symbol: 'UNI', name: 'Uniswap', price: 8.456, change24h: -3.45, volume24h: 150000000, marketCap: 5100000000, logo: '🦄' }
];

export default function CoinPriceMonitor(): JSX.Element {
  const [coins, setCoins] = useState<CoinData[]>(TOP_COINS);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'price' | 'change24h' | 'volume24h' | 'marketCap'>('marketCap');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Simulate live price updates
  useEffect(() => {
    const interval = setInterval(() => {
      setCoins(prevCoins => 
        prevCoins.map(coin => ({
          ...coin,
          price: coin.price * (1 + (Math.random() - 0.5) * 0.02), // ±1% change
          change24h: coin.change24h + (Math.random() - 0.5) * 0.5,
          volume24h: coin.volume24h * (1 + (Math.random() - 0.5) * 0.1)
        }))
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const filteredCoins = coins.filter(coin => 
    coin.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    coin.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const sortedCoins = [...filteredCoins].sort((a, b) => {
    const aValue = a[sortBy];
    const bValue = b[sortBy];
    const multiplier = sortDirection === 'asc' ? 1 : -1;
    return (aValue - bValue) * multiplier;
  });

  const handleSort = (key: typeof sortBy) => {
    if (sortBy === key) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(key);
      setSortDirection('desc');
    }
  };

  const formatNumber = (num: number): string => {
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
    return `$${num.toFixed(2)}`;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white flex items-center">
          💰 Live Coin Prices
        </h3>
        <div className="flex items-center space-x-3">
          <div className="relative">
            <input
              type="text"
              placeholder="Search coins..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-4 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                         bg-white dark:bg-gray-700 text-gray-800 dark:text-white
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
            />
            <div className="absolute right-3 top-2.5 text-gray-400">🔍</div>
          </div>
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b dark:border-gray-700">
              <th className="text-left py-3 px-2 text-gray-600 dark:text-gray-400 font-medium">Coin</th>
              <th 
                className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium cursor-pointer hover:text-blue-500"
                onClick={() => handleSort('price')}
              >
                Price {sortBy === 'price' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium cursor-pointer hover:text-blue-500"
                onClick={() => handleSort('change24h')}
              >
                24h Change {sortBy === 'change24h' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium cursor-pointer hover:text-blue-500"
                onClick={() => handleSort('volume24h')}
              >
                Volume {sortBy === 'volume24h' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                className="text-right py-3 px-2 text-gray-600 dark:text-gray-400 font-medium cursor-pointer hover:text-blue-500"
                onClick={() => handleSort('marketCap')}
              >
                Market Cap {sortBy === 'marketCap' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedCoins.map((coin) => (
              <tr key={coin.symbol} className="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                <td className="py-4 px-2">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{coin.logo}</span>
                    <div>
                      <div className="font-semibold text-gray-800 dark:text-white">{coin.symbol}</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">{coin.name}</div>
                    </div>
                  </div>
                </td>
                <td className="text-right py-4 px-2">
                  <div className="font-semibold text-gray-800 dark:text-white">
                    {coin.price >= 1 ? `$${coin.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : 
                     `$${coin.price.toFixed(4)}`}
                  </div>
                </td>
                <td className="text-right py-4 px-2">
                  <div className={`font-semibold ${
                    coin.change24h >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  }`}>
                    {coin.change24h >= 0 ? '+' : ''}{coin.change24h.toFixed(2)}%
                  </div>
                </td>
                <td className="text-right py-4 px-2">
                  <div className="text-gray-600 dark:text-gray-300">{formatNumber(coin.volume24h)}</div>
                </td>
                <td className="text-right py-4 px-2">
                  <div className="text-gray-600 dark:text-gray-300">{formatNumber(coin.marketCap)}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {filteredCoins.length === 0 && (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No coins found matching "{searchTerm}"
        </div>
      )}
    </div>
  );
}