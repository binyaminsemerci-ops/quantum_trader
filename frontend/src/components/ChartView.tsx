import { useState } from "react";

type ChartViewProps = {
  title?: string;
};

type TradeRow = {
  date: string;
  pair: string;
  side: "BUY" | "SELL";
  amount: number;
  price: number;
};

export default function ChartView({ title }: ChartViewProps): JSX.Element {
  // Sample data from the user
  const balance = 10000;
  const equity = 10250;
  const riskPercent = 1.5;
  const openTrades = 3;

  const [riskPct, setRiskPct] = useState<number>(riskPercent);
  const [entryPrice, setEntryPrice] = useState<string>("");
  const [stopLoss, setStopLoss] = useState<string>("");

  const trades: TradeRow[] = [
    { date: "2025-09-01", pair: "BTC/USDT", side: "BUY", amount: 0.5, price: 25000 },
    { date: "2025-09-02", pair: "ETH/USDT", side: "SELL", amount: 2, price: 1600 },
    { date: "2025-09-03", pair: "BNB/USDT", side: "BUY", amount: 5, price: 280 },
  ];

  // Format helpers: space-separated thousands for balances, comma-separated for prices
  const fmtSpace = (v: number) => new Intl.NumberFormat('fr-FR').format(v);
  const fmtComma = (v: number) => new Intl.NumberFormat('en-US').format(v);

  const [positionSize, setPositionSize] = useState<string | null>(null);

  const parseNumberInput = (raw: string) => {
    // Accept comma as decimal separator (e.g. "1,5") and thin spaces
    if (!raw) return NaN;
    const cleaned = raw.replace(/\s/g, '').replace(',', '.');
    return Number(cleaned);
  };

  const calculatePositionSize = () => {
    const e = parseNumberInput(entryPrice);
    const s = parseNumberInput(stopLoss);
    if (!isFinite(e) || !isFinite(s) || e === s) return '—';
    const riskPerTrade = (riskPct / 100) * balance;
    const riskPerUnit = Math.abs(e - s);
    const size = riskPerTrade / riskPerUnit;
    return size <= 0 || !isFinite(size) ? '—' : size.toFixed(6);
  };

  return (
    <section className="p-4 bg-slate-800 rounded text-white">
      {title && <h2 className="text-lg font-semibold mb-4">{title}</h2>}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className="p-3 bg-slate-700 rounded">
          <div className="text-sm text-slate-300">Balance</div>
          <div className="text-2xl font-bold">${fmtSpace(balance)}</div>
        </div>
        <div className="p-3 bg-slate-700 rounded">
          <div className="text-sm text-slate-300">Equity</div>
          <div className="text-2xl font-bold">${fmtSpace(equity)}</div>
        </div>
        <div className="p-3 bg-slate-700 rounded">
          <div className="text-sm text-slate-300">Risk %</div>
          <div className="text-2xl font-bold">{riskPercent}%</div>
          <div className="text-xs text-slate-400">Open Trades: {openTrades}</div>
        </div>
      </div>

      <div className="mb-4 p-3 bg-slate-700 rounded">
        <div className="text-sm text-slate-300 mb-2">Performance Chart</div>
        <div className="h-40 bg-slate-600 rounded text-slate-300 p-2">
          {/* Small interactive Line chart using react-chartjs-2 */}
          <ChartPlaceholder />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="p-3 bg-slate-700 rounded">
          <h3 className="text-sm font-semibold mb-2">Risk Monitor</h3>
          <div className="space-y-2 text-sm text-slate-300">
            <div>Balance ($): <span className="font-medium text-white">{fmtSpace(balance)}</span></div>
            <div>
              Risk %
              <input
                className="ml-2 px-2 py-1 rounded bg-slate-800 text-white text-sm"
                type="number"
                value={riskPct}
                onChange={(e) => setRiskPct(Number(e.target.value))}
              />
            </div>
            <div>
              Entry Price
              <input
                className="ml-2 px-2 py-1 rounded bg-slate-800 text-white text-sm"
                value={entryPrice}
                onChange={(e) => setEntryPrice(e.target.value)}
                placeholder="e.g. 25000"
              />
            </div>
            <div>
              Stop Loss
              <input
                className="ml-2 px-2 py-1 rounded bg-slate-800 text-white text-sm"
                value={stopLoss}
                onChange={(e) => setStopLoss(e.target.value)}
                placeholder="e.g. 24500"
              />
            </div>
            <div className="mt-2">
              <button
                className="px-3 py-1 bg-indigo-600 rounded text-white text-sm"
                onClick={() => setPositionSize(calculatePositionSize())}
              >
                Calculate Position Size
              </button>
            </div>
            <div className="mt-2">Position Size: <span className="font-medium">{positionSize ?? '—'}</span></div>
          </div>
        </div>

        <div className="p-3 bg-slate-700 rounded overflow-auto">
          <h3 className="text-sm font-semibold mb-2">Trade History</h3>
          <table className="w-full text-sm">
            <thead className="text-slate-300 text-left">
              <tr>
                <th className="pb-2">Date</th>
                <th className="pb-2">Pair</th>
                <th className="pb-2">Side</th>
                <th className="pb-2">Amount</th>
                <th className="pb-2">Price</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.date} className="border-t border-slate-600">
                  <td className="py-2 text-slate-200">{t.date}</td>
                  <td className="py-2 text-slate-200">{t.pair}</td>
                  <td className={`py-2 font-medium ${t.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>{t.side}</td>
                  <td className="py-2 text-slate-200">{t.amount}</td>
                  <td className="py-2 text-slate-200">{fmtComma(t.price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

// --- Chart component ---
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function ChartPlaceholder(): JSX.Element {
  const labels = ['2025-08-28','2025-08-29','2025-08-30','2025-08-31','2025-09-01','2025-09-02','2025-09-03'];
  const data = {
    labels,
    datasets: [
      {
        label: 'Equity',
        data: [9800, 9900, 9950, 10050, 10000, 10150, 10250],
        borderColor: 'rgba(99,102,241,1)',
        backgroundColor: 'rgba(99,102,241,0.3)',
        tension: 0.2,
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: { ticks: { color: '#cbd5e1' } },
      y: { ticks: { color: '#cbd5e1' } },
    },
  } as const;

  return <Line data={data} options={options} />;
}
