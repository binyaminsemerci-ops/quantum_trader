type Trade = { date?: string; pair?: string; side?: 'BUY' | 'SELL' | string; amount?: number | string; price?: number | string };

export default function TradeTable({ trades }: { trades?: Trade[] }): JSX.Element {
  const effective = trades || [];

  return (
    <div className="p-6 bg-white shadow rounded-lg overflow-x-auto">
      <h2 className="text-lg font-semibold text-gray-700 mb-4">Trade History</h2>
      <table className="min-w-full border border-gray-200">
        <thead>
          <tr className="bg-gray-100 text-gray-700">
            <th className="py-2 px-4 border">Date</th>
            <th className="py-2 px-4 border">Pair</th>
            <th className="py-2 px-4 border">Side</th>
            <th className="py-2 px-4 border">Amount</th>
            <th className="py-2 px-4 border">Price</th>
          </tr>
        </thead>
        <tbody>
          {effective.map((trade: any, index) => (
            <tr key={index} className="text-gray-700 text-center">
              <td className="py-2 px-4 border">{trade.timestamp || trade.date || '—'}</td>
              <td className="py-2 px-4 border">{trade.symbol || trade.pair || '—'}</td>
              <td className={`py-2 px-4 border font-semibold ${trade.side === 'BUY' ? 'text-green-600' : 'text-red-600'}`}>{trade.side}</td>
              <td className="py-2 px-4 border">{trade.qty ?? trade.amount ?? '—'}</td>
              <td className="py-2 px-4 border">{trade.price ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
