import { useState } from "react";
import InsightCard from "../components/InsightCard";
import { safeNum, safePct } from '../utils/formatters';

interface TimelineEvent {
  time: string;
  price: number;
  event: string;
  detail: string;
}

interface ReplayData {
  trade_id: number;
  symbol: string;
  direction: string;
  entry_price: number;
  exit_price?: number;
  pnl?: number;
  tp: number;
  sl: number;
  trailing?: number;
  trailing_stop?: number;
  confidence: number;
  model: string;
  timeline: TimelineEvent[];
  features: any;
  policy_state: any;
  exit_reason: string;
  timestamp: string;
}

export default function Replay() {
  const [id, setId] = useState("");
  const [data, setData] = useState<ReplayData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const apiUrl = (import.meta as any).env?.VITE_API_URL || "http://localhost:8026";

  async function fetchReplay() {
    if (!id) {
      setError("Please enter a Trade ID");
      return;
    }
    
    setLoading(true);
    setError("");
    
    try {
      const res = await fetch(`${apiUrl}/replay/${id}`);
      if (!res.ok) {
        throw new Error(`Trade #${id} not found`);
      }
      const json = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  function getEventIcon(event: string) {
    switch (event) {
      case "entry": return "🚀";
      case "tp_target": return "🎯";
      case "sl_threshold": return "🛑";
      case "trailing_active": return "📊";
      case "exit": return "✅";
      default: return "•";
    }
  }

  function getEventColor(event: string) {
    switch (event) {
      case "entry": return "text-blue-400";
      case "tp_target": return "text-green-400";
      case "sl_threshold": return "text-red-400";
      case "trailing_active": return "text-yellow-400";
      case "exit": return "text-purple-400";
      default: return "text-gray-400";
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Trade Replay</h1>

      {/* Search Input */}
      <div className="flex gap-4 p-4 bg-gray-900 border border-gray-800 rounded-lg">
        <input
          type="number"
          value={id}
          onChange={(e) => setId(e.target.value)}
          placeholder="Enter Trade ID (e.g., 1)"
          className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded text-white"
          onKeyPress={(e) => e.key === "Enter" && fetchReplay()}
        />
        <button
          onClick={fetchReplay}
          disabled={loading}
          className="px-6 py-2 bg-green-700 hover:bg-green-600 disabled:bg-gray-700 text-white rounded font-semibold"
        >
          {loading ? "Loading..." : "Replay Trade"}
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* Replay Data */}
      {data && (
        <>
          {/* Trade Overview Cards */}
          <div className="grid gap-4 md:grid-cols-4">
            <InsightCard 
              title={`Trade #${data.trade_id}`}
              value={data.symbol}
              icon={data.direction === "BUY" ? "📈" : "📉"}
            />
            <InsightCard 
              title="Entry Price" 
              value={`$${safeNum(data.entry_price)}`}
            />
            <InsightCard 
              title="Confidence" 
              value={safePct(data.confidence, 0)}
              change={(data.confidence || 0) * 100}
              trend="up"
            />
            <InsightCard 
              title="Model" 
              value={data.model}
              icon="🤖"
            />
          </div>

          {/* Trade Details */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* Timeline */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-4">Timeline</h3>
              <div className="space-y-3">
                {data.timeline.map((event, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <span className="text-2xl">{getEventIcon(event.event)}</span>
                    <div className="flex-1">
                      <div className="flex justify-between items-start">
                        <span className={`font-semibold ${getEventColor(event.event)}`}>
                          {event.event.replace(/_/g, " ").toUpperCase()}
                        </span>
                        <span className="text-white font-mono">${safeNum(event.price)}</span>
                      </div>
                      <p className="text-sm text-gray-400 mt-1">{event.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Trade Metrics */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-4">Trade Metrics</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Direction:</span>
                  <span className={`font-semibold ${
                    data.direction === "BUY" ? "text-green-400" : "text-red-400"
                  }`}>
                    {data.direction}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Take Profit:</span>
                  <span className="text-green-400 font-mono">${safeNum(data.tp)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Stop Loss:</span>
                  <span className="text-red-400 font-mono">${safeNum(data.sl)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Trailing Stop:</span>
                  <span className="text-yellow-400 font-mono">${safeNum(data.trailing_stop)}</span>
                </div>
                {data.exit_price && (
                  <>
                    <div className="flex justify-between border-t border-gray-800 pt-3">
                      <span className="text-gray-400">Exit Price:</span>
                      <span className="text-white font-mono">${safeNum(data.exit_price)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">PnL:</span>
                      <span className={`font-bold ${
                        data.pnl && data.pnl >= 0 ? "text-green-400" : "text-red-400"
                      }`}>
                        ${safeNum(data.pnl)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Exit Reason:</span>
                      <span className="text-purple-400">{data.exit_reason}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Features & Policy State */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-4">AI Features</h3>
              <pre className="text-sm text-gray-300 overflow-auto">
                {JSON.stringify(data.features, null, 2)}
              </pre>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-4">Policy State</h3>
              <pre className="text-sm text-gray-300 overflow-auto">
                {JSON.stringify(data.policy_state, null, 2)}
              </pre>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
