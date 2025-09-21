import React from "react";
import { useDashboardData } from "../hooks/useDashboardData";
import type { StatSummary } from "../types";

export default function AnalyticsCards(): JSX.Element | null {
  const { stats } = useDashboardData();

  if (!stats) return null; // fallback safety

  const safePercent = (n: unknown) => {
    if (typeof n === 'number' && Number.isFinite(n)) return String(n);
    return '—';
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
        <h3 className="text-lg font-semibold">💰 Profit</h3>
  <p className="text-xl">{safePercent((stats as StatSummary).profit)}%</p>
      </div>
      <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
        <h3 className="text-lg font-semibold">✅ Win Rate</h3>
  <p className="text-xl">{safePercent((stats as StatSummary).winRate)}%</p>
      </div>
    </div>
  );
}
