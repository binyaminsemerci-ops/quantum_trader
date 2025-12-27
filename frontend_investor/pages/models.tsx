// pages/models.tsx
import { useEffect, useState } from 'react';
import InvestorNavbar from '@/components/InvestorNavbar';
import LoadingSpinner from '@/components/LoadingSpinner';
import { useAuth } from '@/hooks/useAuth';

interface AIModel {
  name: string;
  weight: number;
  error?: number;
  latency?: number;
  status?: string;
}

export default function Models() {
  const [models, setModels] = useState<AIModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { getToken } = useAuth();

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.quantumfond.com';
      const token = getToken();
      
      const response = await fetch(`${apiUrl}/ai/models`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error('Failed to fetch models');
      
      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      console.error('Models fetch error:', err);
      setError('Failed to load AI model data');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status?: string) => {
    switch (status?.toUpperCase()) {
      case 'ACTIVE':
        return 'bg-green-900/30 text-green-400';
      case 'TRAINING':
        return 'bg-yellow-900/30 text-yellow-400';
      case 'DISABLED':
        return 'bg-red-900/30 text-red-400';
      default:
        return 'bg-quantum-card text-quantum-muted';
    }
  };

  return (
    <div className="min-h-screen bg-quantum-bg">
      <InvestorNavbar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-quantum-text mb-2">AI Model Insights</h1>
          <p className="text-quantum-muted">Ensemble model performance and configuration</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {loading ? (
          <LoadingSpinner />
        ) : (
          <>
            {/* Models Overview */}
            <div className="bg-quantum-card border border-quantum-border rounded-lg p-6 mb-6">
              <h3 className="text-lg font-semibold text-quantum-text mb-4">
                🤖 Ensemble Configuration
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-quantum-accent">{models.length}</div>
                  <div className="text-sm text-quantum-muted">Active Models</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">
                    {models.filter(m => m.status === 'ACTIVE').length}
                  </div>
                  <div className="text-sm text-quantum-muted">Online</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-quantum-text">
                    {models.reduce((sum, m) => sum + (m.weight || 0), 0).toFixed(2)}
                  </div>
                  <div className="text-sm text-quantum-muted">Total Weight</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-quantum-text">
                    {models.length > 0 
                      ? (models.reduce((sum, m) => sum + (m.latency || 0), 0) / models.length).toFixed(0)
                      : '0'}ms
                  </div>
                  <div className="text-sm text-quantum-muted">Avg Latency</div>
                </div>
              </div>
            </div>

            {/* Models Table */}
            <div className="bg-quantum-card border border-quantum-border rounded-lg overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-quantum-dark border-b border-quantum-border">
                    <tr>
                      <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                        Model Name
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                        Weight
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                        Error Rate
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-quantum-muted uppercase tracking-wider">
                        Latency
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-quantum-border">
                    {models.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="px-6 py-8 text-center text-quantum-muted">
                          No model data available
                        </td>
                      </tr>
                    ) : (
                      models.map((model, idx) => (
                        <tr key={idx} className="hover:bg-quantum-dark transition">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <span className="text-quantum-text font-medium">{model.name}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(model.status)}`}>
                              {model.status || 'ACTIVE'}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-quantum-text">
                            <div className="flex items-center">
                              <div className="flex-1 bg-quantum-dark rounded-full h-2 mr-3 max-w-[100px]">
                                <div 
                                  className="bg-quantum-accent h-2 rounded-full" 
                                  style={{ width: `${(model.weight || 0) * 100}%` }}
                                />
                              </div>
                              <span className="text-sm">{((model.weight || 0) * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-quantum-text">
                            {model.error !== undefined ? `${(model.error * 100).toFixed(2)}%` : 'N/A'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-quantum-text">
                            {model.latency !== undefined ? `${model.latency.toFixed(0)}ms` : 'N/A'}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Model Architecture Info */}
            <div className="mt-6 bg-quantum-card border border-quantum-border rounded-lg p-6">
              <h3 className="text-lg font-semibold text-quantum-text mb-4">
                🧠 Model Architecture
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <h4 className="text-quantum-accent font-medium mb-2">Ensemble Strategy</h4>
                  <p className="text-quantum-muted">
                    Weighted voting system combining predictions from multiple specialized models for robust decision-making.
                  </p>
                </div>
                <div>
                  <h4 className="text-quantum-accent font-medium mb-2">Continuous Learning</h4>
                  <p className="text-quantum-muted">
                    Models are continuously retrained on live market data to adapt to changing conditions.
                  </p>
                </div>
                <div>
                  <h4 className="text-quantum-accent font-medium mb-2">Risk Integration</h4>
                  <p className="text-quantum-muted">
                    AI predictions are filtered through Risk Brain for position sizing and exposure control.
                  </p>
                </div>
                <div>
                  <h4 className="text-quantum-accent font-medium mb-2">Performance Tracking</h4>
                  <p className="text-quantum-muted">
                    Each model's performance is monitored in real-time with automatic weight adjustment.
                  </p>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
