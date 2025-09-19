export type OHLCV = {
  timestamp?: string | number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
  [key: string]: unknown;
};

export type Signal = {
  id?: string | number;
  _id?: string | number;
  symbol?: string | null;
  sym?: string | null;
  signal?: string | null;
  signal_type?: string | null;
  confidence?: number | null;
  confidence_score?: number | null;
  timestamp?: string | number | null;
  executed?: boolean | null;
  [key: string]: unknown;
};

export type Trade = {
  trade_id?: string | number;
  id?: string | number;
  symbol?: string;
  side?: 'BUY' | 'SELL' | string;
  quantity?: number | string;
  price?: number | string;
  timestamp?: string | number;
  [key: string]: unknown;
};

export type ApiResponse<T = any> = {
  data?: T;
  error?: string | null;
};
