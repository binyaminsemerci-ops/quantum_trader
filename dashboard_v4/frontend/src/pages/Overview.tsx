import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';
import TrendChart from '../components/TrendChart';
import EventFeed from '../components/EventFeed';
import ControlPanel from '../components/ControlPanel';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface OverviewData {
  portfolio: { pnl: number; exposure: number; positions: number };
  ai: { accuracy: number; sharpe: number };
  risk: { var: number; regime: string };
  system: { cpu: number; ram: number; containers: number };
}

export default function Overview() {
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [trendData, setTrendData] = useState<any[]>([]);

  useEffect(() => {
    const fetchOverview = async () => {
      try {
        const [portfolio, ai, risk, system] = await Promise.all([
          fetch(`${API_BASE_URL}/portfolio/status`).then(r => r.json()),
          fetch(`${API_BASE_URL}/ai/insights`).then(r => r.json()),
          fetch(`${API_BASE_URL}/risk/metrics`).then(r => r.json()),
          fetch(`${API_BASE_URL}/system/health`).then(r => r.json())
        ]);
        const newData = {
          portfolio: {
            pnl: portfolio?.pnl ?? 0,
            exposure: portfolio?.exposure ?? 0,
            positions: portfolio?.positions ?? 0
          },
          ai: {
            accuracy: ai?.accuracy ?? 0.5,
            sharpe: ai?.sharpe ?? 0
          },
          risk: {
            var: risk?.var_95 ?? 0,
            regime: risk?.regime ?? 'UNKNOWN'
          },
          system: {
            cpu: system?.metrics?.cpu ?? 0,
            ram: system?.metrics?.ram ?? 0,
            containers: system?.container_count ?? 0
          }
        };
        setData(newData);
        
        // Add to trend data with timestamp
        setTrendData(prev => [...prev.slice(-20), {
          timestamp: Date.now() / 1000,
          accuracy: ai?.accuracy ?? 0.5,
          cpu: system?.metrics?.cpu ?? 0,
          ram: system?.metrics?.ram ?? 0
        }]);
        
        setLoading(false);
      } catch (err) {
        console.error('Failed to load overview:', err);
        setLoading(false);
      }
    };

    fetchOverview();
    const interval = setInterval(fetchOverview, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading system overview...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400">Failed to load overview data</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-green-400">System Overview</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <InsightCard 
          title="AI Accuracy" 
          value={`${((data.ai?.accuracy ?? 0.5) * 100).toFixed(1)}%`} 
          subtitle="Model prediction accuracy"
          color="text-green-400"
        />
        <InsightCard 
          title="CPU Usage" 
          value={`${(data.system?.cpu ?? 0).toFixed(1)}%`} 
          subtitle={`${data.system?.containers ?? 0} containers running`}
          color="text-yellow-400"
        />
        <InsightCard 
          title="RAM Usage" 
          value={`${(data.system?.ram ?? 0).toFixed(1)}%`} 
          subtitle="Memory consumption"
          color="text-blue-400"
        />
        <InsightCard 
          title="Portfolio PnL" 
          value={`$${(data.portfolio?.pnl ?? 0).toLocaleString()}`} 
          subtitle={`${data.portfolio?.positions ?? 0} active positions`}
          color="text-green-400"
        />
        <InsightCard 
          title="Market Regime" 
          value={data.risk?.regime ?? 'UNKNOWN'} 
          subtitle={`VaR: ${((data.risk?.var ?? 0) * 100).toFixed(2)}%`}
          color="text-purple-400"
        />
        <InsightCard 
          title="Sharpe Ratio" 
          value={(data.ai?.sharpe ?? 0).toFixed(3)} 
          subtitle="Risk-adjusted returns"
          color="text-cyan-400"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 p-4 rounded-xl">
          <h2 className="text-xl font-semibold text-white mb-4">AI Accuracy Trend</h2>
          <TrendChart data={trendData} dataKey="accuracy" color="#00FF99" />
        </div>

        <div className="bg-gray-800 p-4 rounded-xl">
          <h2 className="text-xl font-semibold text-white mb-4">CPU Usage Trend</h2>
          <TrendChart data={trendData} dataKey="cpu" color="#FBBF24" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <EventFeed />
        
        <div className="space-y-4">
          <div className="bg-gray-800 p-4 rounded-xl">
            <h2 className="text-xl font-semibold text-white mb-4">System Control</h2>
            <ControlPanel />
          </div>
          
          <div className="bg-gray-800 p-4 rounded-xl">
            <h2 className="text-xl font-semibold text-white mb-4">Quick Stats</h2>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Active Positions:</span>
                <span className="text-white font-bold">{data.portfolio?.positions ?? 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Exposure:</span>
                <span className="text-white font-bold">{((data.portfolio?.exposure ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Containers:</span>
                <span className="text-white font-bold">{data.system?.containers ?? 0}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
