import { useState } from 'react';

export default function RiskMonitor(): JSX.Element {
  const [balance, setBalance] = useState<number>(10000);
  const [riskPercent, setRiskPercent] = useState<number>(1.0);
  const [entryPrice, setEntryPrice] = useState<string>('');
  const [stopLoss, setStopLoss] = useState<string>('');
  const [positionSize, setPositionSize] = useState<number | null>(null);

  // sample derived values shown in the panel
  const equity = 10250;
  const openTrades = 3;

  const calculateRisk = () => {
  if (entryPrice === '' || stopLoss === '') return;

  const riskAmount = balance * (riskPercent / 100);
  const perUnitLoss = Number.parseFloat(entryPrice) - Number.parseFloat(stopLoss);

    if (!Number.isFinite(perUnitLoss) || perUnitLoss <= 0) {
      setPositionSize(0);
      return;
    }

    const size = Math.floor(riskAmount / perUnitLoss);
    setPositionSize(Number.isFinite(size) ? size : 0);
  };

  return (
    <div className="p-6 bg-white shadow rounded-lg">
      <h2 className="text-lg font-semibold text-gray-700 mb-4">Risk Monitor</h2>
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-4 mb-2">
          <div className="p-3 bg-gray-50 rounded">
            <div className="text-xs text-gray-500">Balance</div>
            <div className="text-lg font-semibold">${balance.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <div className="text-xs text-gray-500">Equity</div>
            <div className="text-lg font-semibold">${equity.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <div className="text-xs text-gray-500">Risk %</div>
            <div className="text-lg font-semibold">{riskPercent}%</div>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <div className="text-xs text-gray-500">Open Trades</div>
            <div className="text-lg font-semibold">{openTrades}</div>
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600">Balance ($)</label>
          <input
            type="number"
            className="w-full border rounded p-2"
            value={balance}
            onChange={(e) => setBalance(parseFloat(e.target.value))}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600">Risk %</label>
          <input
            type="number"
            className="w-full border rounded p-2"
            value={riskPercent}
            onChange={(e) => setRiskPercent(parseFloat(e.target.value))}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600">Entry Price</label>
          <input
            type="number"
            className="w-full border rounded p-2"
            value={entryPrice === '' ? '' : String(entryPrice)}
            onChange={(e) => setEntryPrice(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600">Stop Loss</label>
          <input
            type="number"
            className="w-full border rounded p-2"
            value={stopLoss === '' ? '' : String(stopLoss)}
            onChange={(e) => setStopLoss(e.target.value)}
          />
        </div>
        <button
          onClick={calculateRisk}
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
        >
          Calculate Position Size
        </button>
        {positionSize !== null && (
          <div className="mt-4 p-3 bg-gray-100 rounded text-center">
            <p className="text-gray-700 font-medium">
              Position Size: <span className="font-bold">{positionSize} units</span>
            </p>
          </div>
        )}
        {/* Trade history sample table */}
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-600 mb-2">Trade History</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm text-left">
              <thead>
                <tr className="bg-gray-100">
                  <th className="px-3 py-2">Date</th>
                  <th className="px-3 py-2">Pair</th>
                  <th className="px-3 py-2">Side</th>
                  <th className="px-3 py-2">Amount</th>
                  <th className="px-3 py-2">Price</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="px-3 py-2">2025-09-01</td>
                  <td className="px-3 py-2">BTC/USDT</td>
                  <td className="px-3 py-2">BUY</td>
                  <td className="px-3 py-2">0.5</td>
                  <td className="px-3 py-2">25,000</td>
                </tr>
                <tr className="bg-white">
                  <td className="px-3 py-2">2025-09-02</td>
                  <td className="px-3 py-2">ETH/USDT</td>
                  <td className="px-3 py-2">SELL</td>
                  <td className="px-3 py-2">2</td>
                  <td className="px-3 py-2">1,600</td>
                </tr>
                <tr>
                  <td className="px-3 py-2">2025-09-03</td>
                  <td className="px-3 py-2">BNB/USDT</td>
                  <td className="px-3 py-2">BUY</td>
                  <td className="px-3 py-2">5</td>
                  <td className="px-3 py-2">280</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
