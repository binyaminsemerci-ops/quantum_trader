import { describe, it, expect } from 'vitest';
import { safeParse, extractToastFromPayload } from '../utils/ws';

describe('ws helpers', () => {
  it('safeParse returns object for valid JSON string', () => {
    const s = JSON.stringify({ a: 1 });
    expect(safeParse(s)).toEqual({ a: 1 });
  });

  it('safeParse returns undefined for invalid JSON', () => {
    expect(safeParse('not-json')).toBeUndefined();
  });

  it('extractToastFromPayload returns null when no logs', () => {
    expect(extractToastFromPayload({})).toBeNull();
  });

  it('extractToastFromPayload returns toast for logs array', () => {
    const payload = { logs: [{ status: 'accepted', symbol: 'BTCUSD', side: 'BUY', qty: 1, price: 100 }] };
    const t = extractToastFromPayload(payload);
    expect(t).not.toBeNull();
    expect(t?.type).toBe('success');
    expect((t as any).message).toContain('BTCUSD');
  });
});
