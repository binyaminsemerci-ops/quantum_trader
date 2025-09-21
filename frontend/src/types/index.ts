export type ApiResponse<T = unknown> = { data?: T; error?: string };

export type Trade = { id?: string | number };
export type StatSummary = Record<string, any>;
export type OHLCV = { timestamp?: string; open?: number; high?: number; low?: number; close?: number; volume?: number };

// Balance shapes returned by backend routes (backend/routes/binance.py)
export type SpotBalance = { asset: string; free: number };
export type FuturesBalance = { asset: string; balance: number };

export type Balance = SpotBalance | FuturesBalance;
