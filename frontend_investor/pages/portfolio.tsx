// pages/portfolio.tsx
import { useEffect, useState } from 'react';
import InvestorNavbar from '@/components/InvestorNavbar';
import LoadingSpinner from '@/components/LoadingSpinner';
import { useAuth } from '@/hooks/useAuth';

interface Position {
  id: number;
  symbol: string;
  direction: string;
  entry_price: number;
  current_price?: number;
  pnl?: number;
  tp?: number;
  sl?: number;
  confidence?: number;
}

export default function Portfolio() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { getToken } = useAuth();

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const fetchPortfolio = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.quantumfond.com';
      const token = getToken();
      
      const response = await fetch(`${apiUrl}/trades/open`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error('Failed to fetch portfolio');
      
      const data = await response.json();
      setPositions(data.positions || []);
    } catch (err) {
      console.error('Portfolio fetch error:', err);
      setError('Failed to load portfolio data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-quantum-bg">
      <InvestorNavbar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-quantum-text mb-2">Portfolio Positions</h1>
          <p className="text-quantum-muted">Current active trading positions</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {loading ? (
          <LoadingSpinner />
        ) : (
          <div className="bg-quantum-card border border-quantum-border rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-quantum-dark border-b border-quantum-border">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                      Symbol
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                      Direction
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                      Entry
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                      Current
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                      P&L
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                      TP / SL
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                      Confidence
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-quantum-border">
                  {positions.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-6 py-8 text-center text-quantum-muted">
                        No active positions
                      </td>
                    </tr>
                  ) : (
                    positions.map((pos) => (
                      <tr key={pos.id} className="hover:bg-quantum-dark transition">
                        <td className="px-6 py-4 whitespace-nowrap font-medium text-quantum-text">
                          {pos.symbol}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            pos.direction === 'BUY' 
                              ? 'bg-green-900/30 text-green-400'
                              : 'bg-red-900/30 text-red-400'
                          }`}>
                            {pos.direction}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-quantum-text">
                          ${pos.entry_price.toFixed(2)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-quantum-text">
                          ${pos.current_price?.toFixed(2) || 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={pos.pnl && pos.pnl > 0 ? 'text-green-400' : 'text-red-400'}>
                            {pos.pnl ? `$${pos.pnl.toFixed(2)}` : 'N/A'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-quantum-muted">
                          {pos.tp ? `$${pos.tp.toFixed(2)}` : 'N/A'} / {pos.sl ? `$${pos.sl.toFixed(2)}` : 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-quantum-text">
                          {pos.confidence ? `${(pos.confidence * 100).toFixed(0)}%` : 'N/A'}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
