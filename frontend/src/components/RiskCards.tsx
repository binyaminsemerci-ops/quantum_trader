import React from "react";
import { useDashboardData, StatSummary } from "../hooks/useDashboardData";

export default function RiskCards(): JSX.Element | null {
  const { stats } = useDashboardData();

  if (!stats) return null; // fallback safety

  const safe = (n: unknown) => (typeof n === 'number' && Number.isFinite(n) ? String(n) : '—');

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
        <h3 className="text-lg font-semibold">📉 Max Drawdown</h3>
        <p className="text-xl">{safe((stats as StatSummary).maxDrawdown)}%</p>
      </div>
      <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
        <h3 className="text-lg font-semibold">📊 Sharpe Ratio</h3>
        <p className="text-xl">{safe((stats as StatSummary).sharpe)}</p>
      </div>
    </div>
  );
}
