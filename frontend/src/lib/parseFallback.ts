import type { Trade, OHLCV } from '../types';

export function parseTradesPayload(payload: unknown): Trade[] {
  if (Array.isArray(payload)) return payload as Trade[];
  if (payload && typeof payload === 'object' && 'trades' in (payload as any) && Array.isArray((payload as any).trades)) {
    return (payload as any).trades as Trade[];
  }
  return [];
}

export function parseLogsPayload(payload: unknown): Record<string, unknown>[] {
  if (Array.isArray(payload)) return payload as Record<string, unknown>[];
  if (payload && typeof payload === 'object' && 'logs' in (payload as any) && Array.isArray((payload as any).logs)) {
    return (payload as any).logs as Record<string, unknown>[];
  }
  return [];
}

export function parseChartPayload(payload: unknown): OHLCV[] {
  if (Array.isArray(payload)) return payload as OHLCV[];
  return [];
}

export default { parseTradesPayload, parseLogsPayload, parseChartPayload };
