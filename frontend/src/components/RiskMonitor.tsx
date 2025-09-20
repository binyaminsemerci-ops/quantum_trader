import React, { useState } from "react";

export default function RiskMonitor(): JSX.Element {
  const [balance, setBalance] = useState<number>(10000);
  const [riskPercent, setRiskPercent] = useState<number>(1.0);
  const [entryPrice, setEntryPrice] = useState<number | null>(null);
  const [stopLoss, setStopLoss] = useState<number | null>(null);
  const [positionSize, setPositionSize] = useState<number | null>(null);

  const calculateRisk = () => {
    if (entryPrice == null || stopLoss == null) return;

    const riskAmount = balance * (riskPercent / 100);
    const perUnitLoss = (entryPrice as number) - (stopLoss as number);

    if (perUnitLoss <= 0) {
      setPositionSize(0);
      return;
    }

    const size = Math.floor(riskAmount / perUnitLoss);
    setPositionSize(size);
  };

  return (
    <div className="p-6 bg-white shadow rounded-lg">
      <h2 className="text-lg font-semibold text-gray-700 mb-4">Risk Monitor</h2>
      <div className="space-y-3">
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
            value={entryPrice ?? ''}
            onChange={(e) => setEntryPrice(parseFloat(e.target.value))}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600">Stop Loss</label>
          <input
            type="number"
            className="w-full border rounded p-2"
            value={stopLoss ?? ''}
            onChange={(e) => setStopLoss(parseFloat(e.target.value))}
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
      </div>
    </div>
  );
}
