import React from "react";
import type { Trade } from "../types";

export type TradeLogProps = {
  trades?: Trade[] | null;
};

export default function TradeLog({ trades }: TradeLogProps): JSX.Element {
  const safeTrades = Array.isArray(trades) ? trades : [];
  return (
    <div className="p-4 border rounded bg-gray-50">
      <h2 className="text-xl font-bold mb-2">Trade Log</h2>
      {safeTrades.length === 0 ? (
        <p>No trades yet...</p>
      ) : (
        <ul>
          {safeTrades.map((t) => (
            <li key={String(t.trade_id ?? t.id ?? Math.random())}>{t.symbol ?? '—'} {t.side ?? ''} @{t.price ?? 'market'}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
