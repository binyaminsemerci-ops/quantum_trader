import { vi, describe, it, expect } from 'vitest';
import { api } from '../utils/api';

describe('api.post params handling', () => {
  it('appends params to the query string when provided', async () => {
    vi.restoreAllMocks();
    // capture the request URL by mocking fetch
    let calledUrl = '';
    // @ts-ignore
    global.fetch = vi.fn((url) => { calledUrl = String(url); return Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true }) }); });

    await api.post('/test?existing=1', null, { params: { foo: 'bar', q: '1' } });
    expect(calledUrl).toMatch(/existing=1/);
    expect(calledUrl).toMatch(/foo=bar/);
    expect(calledUrl).toMatch(/q=1/);
  });
});
