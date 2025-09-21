export type ApiResponse<T = unknown> = { data?: T; error?: string };

export type StatSummary = { total_trades?: number; pnl?: number };
export type OHLCV = number | { timestamp?: string; open?: number; high?: number; low?: number; close?: number; volume?: number };
export type Trade = { id?: string | number; symbol?: string; side?: 'BUY' | 'SELL' };

// Balance shapes returned by backend routes (backend/routes/binance.py)
export type SpotBalance = { asset: string; free: number };
export type FuturesBalance = { asset: string; balance: number };

export type Balance = SpotBalance | FuturesBalance;
