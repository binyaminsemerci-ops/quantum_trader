import { useEquity } from "../store/tradeStore";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import type { OHLCV } from "../types";

type EquityHookResult = {
  data?: { equity_curve?: Array<Record<string, any>> } | null;
  isLoading?: boolean;
  isError?: boolean;
};

export default function EquityChart(): JSX.Element {
  const { data, isLoading, isError } = useEquity() as EquityHookResult;

  if (isLoading) return <div>Loading equity...</div>;
  if (isError) return <div>Failed to load equity curve</div>;

  const chartData = Array.isArray(data?.equity_curve) ? data!.equity_curve! : [];

  return (
    <div className="w-full h-80 border rounded-xl shadow p-4">
      <h2 className="text-xl font-bold mb-2">Equity Curve</h2>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="trade_id" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="balance" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
