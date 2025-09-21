import { describe, test, expect } from 'vitest';
import { calcPositionSize } from './position';

describe('calcPositionSize', () => {
  test('calculates size for numeric inputs', () => {
    const size = calcPositionSize(10000, 1.5, 25000, 24500);
    // risk = 150, riskPerUnit = 500 => size = 0.3
    expect(size).toBeCloseTo(0.3, 6);
  });

  test('parses comma decimal and spaces', () => {
    const size = calcPositionSize(10000, 1, '25 000', '24 500');
    // risk = 100, per unit = 500 => 0.2
    expect(size).toBeCloseTo(0.2, 6);
  });

  test('returns null for invalid inputs', () => {
    expect(calcPositionSize(10000, 1, '', '')).toBeNull();
    expect(calcPositionSize(10000, 1, 'abc', '123')).toBeNull();
  });
});
