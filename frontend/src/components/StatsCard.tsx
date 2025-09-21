import React from "react";

export type StatsCardProps = {
  title: string;
  value: string | number | null | undefined;
};

function StatsCard({ title, value }: StatsCardProps) {
  return (
    <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
      <h2 className="font-bold">{title}</h2>
      <p>{value == null ? '-' : String(value)}</p>
    </div>
  );
}

export default function StatsCards(): JSX.Element {
  return (
    <div className="grid grid-cols-2 gap-4">
      <StatsCard title="Profit" value={"+1234 USD"} />
      <StatsCard title="Trades" value={56} />
    </div>
  );
}
