import { describe, it, beforeEach, afterEach, expect, vi } from 'vitest';
import { fetchRecentPrices } from '../../src/api/prices';

describe('fetchRecentPrices', () => {
  beforeEach(() => {
    // @ts-ignore
    global.fetch = vi.fn();
  });

  afterEach(() => {
    // @ts-ignore
    global.fetch = undefined;
  });

  it('returns parsed candles when API responds', async () => {
    const mockData = [
      { time: '2025-09-24T00:00:00Z', open: 100, high: 101, low: 99, close: 100, volume: 10 },
    ];
    // @ts-ignore
    global.fetch.mockResolvedValueOnce({ ok: true, json: async () => mockData });

    const candles = await fetchRecentPrices('BTCUSDC', 1);
    expect(candles).toHaveLength(1);
    expect(candles[0].close).toBe(100);
  });

  it('returns fallback candles when fetch fails', async () => {
    // @ts-ignore
    global.fetch.mockRejectedValueOnce(new Error('network'));
    const candles = await fetchRecentPrices('BTCUSDC', 3);
    expect(candles).toHaveLength(3);
  });
});
