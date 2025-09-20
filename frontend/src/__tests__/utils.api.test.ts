import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { api } from '../utils/api';

// We'll test the request behavior by mocking the global fetch.
describe('utils/api request guards', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    // reset mock
    vi.restoreAllMocks();
  });

  afterEach(() => {
    // restore original fetch
    // @ts-ignore
    global.fetch = originalFetch;
  });

  it('returns data when JSON payload is returned', async () => {
    // @ts-ignore
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ x: 1 }) }));
    const res = await api.get<{ x: number }>('/test');
    expect(res.data).toEqual({ x: 1 });
    expect(res.error).toBeUndefined();
  });

  it('returns error when non-ok text response is returned', async () => {
    // @ts-ignore
    global.fetch = vi.fn(() => Promise.resolve({ ok: false, status: 500, text: () => Promise.resolve('Server error') }));
    const res = await api.get('/test');
    expect(res.error).toMatch(/API error 500/);
    expect(res.data).toBeUndefined();
  });

  it('handles invalid JSON gracefully by returning undefined data', async () => {
    // @ts-ignore
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.reject(new Error('invalid json')) }));
    const res = await api.get('/test');
    expect(res.data).toBeUndefined();
    expect(res.error).toBeUndefined();
  });
});
