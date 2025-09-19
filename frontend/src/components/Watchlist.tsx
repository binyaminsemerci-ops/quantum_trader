import React from "react";
import { useDashboardData } from "../hooks/useDashboardData";
import type { DashboardData } from "../hooks/useAutoRefresh";

type WatchEntry = { symbol?: string; price?: string | number };

type WatchlistProps = {
  entries?: WatchEntry[] | null;
};

export default function Watchlist({ entries }: WatchlistProps = {}): JSX.Element | null {
  const { watchlist } = useDashboardData();
  const list = (entries ?? watchlist ?? []) as WatchEntry[];

  if (!list || !Array.isArray(list) || list.length === 0) {
    return (
      <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
        No symbols in watchlist.
      </div>
    );
  }

  return (
    <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
      <h2 className="text-lg font-bold mb-4">👀 Watchlist</h2>
      <ul>
        {list.map((w: WatchEntry, i: number) => (
          <li key={i} className="flex justify-between border-b py-1">
            <span>{w?.symbol ?? '—'}</span>
            <span className="font-mono">{w?.price ?? '—'}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
