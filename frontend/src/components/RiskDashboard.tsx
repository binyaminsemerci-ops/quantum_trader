import React, { useEffect, useState, useMemo } from 'react';
import { Shield, AlertTriangle, BarChart3, Activity, Layers, Target, Zap } from 'lucide-react';

interface ExposureRow {
  symbol: string;
  marketValue: number;
  exposurePct: number;
  unrealizedPnL: number;
}
interface RiskOverview {
  riskScore: number;
  volatility: number; // 0-1
  var95: number; // negative VaR 95
  exposureTotalPct: number;
  leverage: number;
  marginUsed: number;
  drawdownRisk: string;
  correlationRisk: string;
  liquidationPrice: number | null;
  limits: {
    max_daily_loss: number;
    max_drawdown_allowed: number;
    max_symbol_exposure_pct: number;
    max_portfolio_leverage: number;
  };
  exposures: ExposureRow[];
  timestamp: string;
}

const colorBand = (score:number) => score <= 30 ? 'text-green-500' : score <= 60 ? 'text-yellow-500' : 'text-red-500';

const miniStat = (label:string, value:React.ReactNode, icon:React.ReactNode, danger=false) => (
  <div className={`flex items-center gap-3 p-3 rounded border bg-white dark:bg-gray-800 dark:border-gray-700 shadow-sm ${danger ? 'border-red-300 dark:border-red-700' : 'border-gray-200'}`}> 
    <div className="p-2 rounded bg-gray-100 dark:bg-gray-700">{icon}</div>
    <div className="flex-1">
      <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide">{label}</div>
      <div className="text-sm font-semibold text-gray-800 dark:text-gray-100">{value}</div>
    </div>
  </div>
);

export default function RiskDashboard({ refreshMs = 5000 }: { refreshMs?: number }) {
  const [data, setData] = useState<RiskOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      setError(null);
      const res = await fetch('/api/v1/risk/overview');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const json = await res.json();
      setData(json);
    } catch (e:any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, refreshMs);
    return () => clearInterval(id);
  }, [refreshMs]);

  const topExposures = useMemo(() => (data?.exposures || []).slice().sort((a,b)=> b.exposurePct - a.exposurePct).slice(0,5), [data]);

  if (loading) {
    return <div className="h-full flex items-center justify-center text-xs text-gray-500">Laster risiko...</div>;
  }
  if (error) {
    return <div className="h-full flex flex-col items-center justify-center text-xs text-red-500">Feil: {error}</div>;
  }
  if (!data) return null;

  return (
    <div className="flex flex-col h-full gap-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {miniStat('RISK SCORE', <span className={colorBand(data.riskScore)+ ' text-lg'}>{data.riskScore}</span>, <Shield className="w-4 h-4" />)}
        {miniStat('VOLATILITY', (data.volatility*100).toFixed(1)+'%', <Activity className="w-4 h-4" />)}
        {miniStat('MARGIN USED', data.marginUsed.toFixed(1)+'%', <Layers className="w-4 h-4" />, data.marginUsed>70)}
        {miniStat('LEVERAGE', data.leverage.toFixed(2)+'x', <Zap className="w-4 h-4" />, data.leverage>2.5)}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-1 min-h-0">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold">Eksponering</h3>
            <BarChart3 className="w-4 h-4 text-gray-400" />
          </div>
          <div className="space-y-2 overflow-auto custom-scrollbar pr-1">
            {topExposures.map(ex => (
              <div key={ex.symbol} className="flex items-center justify-between border-b last:border-b-0 pb-1">
                <div className="text-xs font-medium">{ex.symbol}</div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-gray-500">{ex.exposurePct.toFixed(2)}%</span>
                  <span className={ex.unrealizedPnL>=0 ? 'text-green-600' : 'text-red-500'}>{ex.unrealizedPnL.toFixed(2)}</span>
                </div>
              </div>
            ))}
            {!topExposures.length && <div className="text-xs text-gray-500">Ingen posisjoner</div>}
          </div>
          <div className="mt-auto pt-2 text-[11px] text-gray-500">Total eksponering: {data.exposureTotalPct.toFixed(1)}%</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold">Risikoindikatorer</h3>
            <AlertTriangle className="w-4 h-4 text-gray-400" />
          </div>
          <ul className="space-y-1 text-xs">
            <li className="flex justify-between"><span>Drawdown Risk</span><span className="font-semibold">{data.drawdownRisk}</span></li>
            <li className="flex justify-between"><span>Correlation Risk</span><span className="font-semibold">{data.correlationRisk}</span></li>
            <li className="flex justify-between"><span>VaR 95%</span><span className="font-semibold text-red-500">{data.var95}</span></li>
            <li className="flex justify-between"><span>Limits: Max DD</span><span className="font-semibold">{(data.limits.max_drawdown_allowed*100).toFixed(1)}%</span></li>
            <li className="flex justify-between"><span>Limits: Max Sym Exp</span><span className="font-semibold">{data.limits.max_symbol_exposure_pct}%</span></li>
            <li className="flex justify-between"><span>Limits: Max Lev</span><span className="font-semibold">{data.limits.max_portfolio_leverage}x</span></li>
          </ul>
          <div className="mt-auto text-[10px] text-gray-500 pt-2">Oppdatert {new Date(data.timestamp).toLocaleTimeString()}</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold">Score Breakdown</h3>
            <Target className="w-4 h-4 text-gray-400" />
          </div>
          <div className="flex-1 flex flex-col gap-2">
            <div>
              <div className="flex justify-between text-[11px]"><span>Belastning</span><span>{Math.min(100, data.marginUsed).toFixed(0)}%</span></div>
              {/** width bucketization into tailwind w-* classes */}
              {(() => { const pct = Math.min(100, data.marginUsed); const w = `w-[${pct}%]`; return (
                <div className="h-2 rounded bg-gray-200 dark:bg-gray-700 overflow-hidden"><div className={`h-full bg-indigo-500 ${w}`}/></div>
              ); })()}
            </div>
            <div>
              <div className="flex justify-between text-[11px]"><span>Volatilitet</span><span>{(data.volatility*100).toFixed(1)}%</span></div>
              {(() => { const pct = Math.min(100, data.volatility*140); const w = `w-[${pct}%]`; return (
                <div className="h-2 rounded bg-gray-200 dark:bg-gray-700 overflow-hidden"><div className={`h-full bg-amber-500 ${w}`}/></div>
              ); })()}
            </div>
            <div>
              <div className="flex justify-between text-[11px]"><span>Konsentrasjon</span><span>{topExposures[0]?.exposurePct?.toFixed(1) || 0}%</span></div>
              {(() => { const pct = Math.min(100, (topExposures[0]?.exposurePct||0)); const w = `w-[${pct}%]`; return (
                <div className="h-2 rounded bg-gray-200 dark:bg-gray-700 overflow-hidden"><div className={`h-full bg-pink-500 ${w}`}/></div>
              ); })()}
            </div>
          </div>
          <div className="mt-auto text-[10px] text-gray-500">Heuristisk modell – ikke endelig risikomotor.</div>
        </div>
      </div>

      <div className="text-[10px] text-gray-500 dark:text-gray-400">Beta: Tall er syntetiske estimater. Når ekte trading aktiveres kobles denne mot sanntidseksponering, volatilitet & PnL-historikk.</div>
    </div>
  );
}
