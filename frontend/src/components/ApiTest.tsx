import React, { useState } from "react";
import { api } from "../utils/api";
import type { ApiResponse } from "../types";

export default function ApiTest(): JSX.Element {
  const [result, setResult] = useState<ApiResponse<unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleTest(endpoint: string) {
    setLoading(true);
    setResult(null);
    try {
      // use the compatibility get<T>() which returns ApiResponse<T>
      const res = await api.get<unknown>(endpoint);
      setResult(res);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setResult({ error: message });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-4 bg-white rounded shadow">
      <h2 className="text-lg font-bold mb-4">🧪 API Tester</h2>

      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={() => handleTest("/binance/balance")}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Test Binance Balance
        </button>
        <button
          onClick={() => handleTest("/stats")}
          className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
        >
          Test Stats
        </button>
        <button
          onClick={() => handleTest("/trades")}
          className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
        >
          Test Trades
        </button>
        <button
          onClick={() => handleTest("/chart")}
          className="bg-orange-600 text-white px-4 py-2 rounded hover:bg-orange-700"
        >
          Test Chart
        </button>
        <button
          onClick={() => handleTest("/settings")}
          className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700"
        >
          Test Settings
        </button>
      </div>

      {loading && <p>⏳ Kjører test...</p>}

      {result && (
        <div className="p-2 bg-gray-100 rounded text-sm overflow-x-auto">
          <pre className="whitespace-pre-wrap break-words">
            {result.data ? JSON.stringify(result.data, null, 2) : `Error: ${result.error}`}
          </pre>
        </div>
      )}
    </div>
  );
}
