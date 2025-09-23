export type ApiResponse<T = unknown> = { data?: T; error?: string };

export type Trade = { id?: string | number };
export type StatSummary = Record<string, any>;
export type OHLCV = { timestamp?: string; open?: number; high?: number; low?: number; close?: number; volume?: number };
