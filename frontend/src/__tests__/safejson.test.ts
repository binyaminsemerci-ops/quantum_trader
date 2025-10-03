import { describe, it, expect } from 'vitest';
import { safeJson } from '../utils/api';

function makeResponse(body: unknown, ok = true) {
  return {
    ok,
    async json() {
      if (body === Symbol.for('malformed')) throw new Error('malformed');
      return body;
    },
    async text() {
      return String(body);
    },
  } as unknown as Response;
}

describe('safeJson', () => {
  it('returns parsed object when valid', async () => {
    const res = makeResponse({ a: 1 });
    const out = await safeJson(res);
    expect(out).toEqual({ a: 1 });
  });

  it('returns undefined on parse error', async () => {
    const res = makeResponse(Symbol.for('malformed'));
    const out = await safeJson(res);
    expect(out).toBeUndefined();
  });
});
