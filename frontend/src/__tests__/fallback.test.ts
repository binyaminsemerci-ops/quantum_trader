import { describe, it, expect } from 'vitest';
import { extractWrapperArray } from '../utils/api';

describe('extractWrapperArray', () => {
  it('returns array when payload is array', () => {
    const payload = [{ id: 1 }, { id: 2 }];
    expect(extractWrapperArray(payload, 'trades')).toEqual(payload);
  });

  it('returns named array when payload is wrapper', () => {
    const payload = { trades: [{ id: 3 }], other: [] };
    expect(extractWrapperArray(payload, 'trades')).toEqual([{ id: 3 }]);
  });

  it('returns empty array for malformed payload', () => {
    expect(extractWrapperArray(null, 'trades')).toEqual([]);
    expect(extractWrapperArray({ trades: 'not-an-array' }, 'trades')).toEqual([]);
  });
});
