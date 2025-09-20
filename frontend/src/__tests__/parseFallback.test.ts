import { parseTradesPayload, parseLogsPayload, parseChartPayload } from '../lib/parseFallback';

describe('parseFallback helpers', () => {
  test('parseTradesPayload returns array when given array', () => {
    const input = [{ id: 1, symbol: 'BTC' }];
    expect(parseTradesPayload(input)).toEqual(input);
  });

  test('parseTradesPayload extracts .trades when present', () => {
    const input = { trades: [{ id: 2, symbol: 'ETH' }] };
    expect(parseTradesPayload(input)).toEqual(input.trades);
  });

  test('parseTradesPayload returns empty for malformed payload', () => {
    expect(parseTradesPayload(null)).toEqual([]);
    expect(parseTradesPayload({ foo: 'bar' })).toEqual([]);
  });

  test('parseLogsPayload extracts logs array', () => {
    const input = { logs: [{ msg: 'x' }] };
    expect(parseLogsPayload(input)).toEqual(input.logs);
  });

  test('parseChartPayload returns array for arrays and empty otherwise', () => {
    const c = [{ timestamp: 't', open: 1, high: 2, low: 0, close: 1 }];
    expect(parseChartPayload(c)).toEqual(c);
    expect(parseChartPayload({})).toEqual([]);
  });
});
