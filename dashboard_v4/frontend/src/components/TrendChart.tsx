import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

type Props = {
  data: any[];
  dataKey: string;
  color?: string;
};

export default function TrendChart({ data, dataKey, color = "#00FF99" }: Props) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis 
          dataKey="timestamp" 
          tickFormatter={(t) => new Date(t * 1000).toLocaleTimeString()} 
        />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}
