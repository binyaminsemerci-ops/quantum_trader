import React from "react";

export default function StatsCard({ title, value }) {
  return (
    <div className="p-4 bg-white shadow rounded-lg text-center">
      <h3 className="text-lg font-semibold text-gray-700">{title}</h3>
      <p className="text-2xl font-bold text-gray-900 mt-2">{value}</p>
    </div>
  );
}
