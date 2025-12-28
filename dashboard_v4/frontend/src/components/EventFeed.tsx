import { useEffect, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "wss://app.quantumfond.com/api/events/stream";

export default function EventFeed() {
  const [events, setEvents] = useState<any[]>([]);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    
    ws.onmessage = (e) => {
      setEvents((prev) => [JSON.parse(e.data), ...prev].slice(0, 15));
    };

    ws.onerror = () => {
      console.error("WebSocket connection failed");
    };

    return () => ws.close();
  }, []);

  return (
    <div className="bg-gray-900 p-3 rounded-lg h-80 overflow-y-auto">
      <h3 className="text-lg mb-2 text-green-400 font-semibold">Live Event Feed</h3>
      {events.length === 0 && (
        <div className="text-gray-500 text-sm">Waiting for events...</div>
      )}
      {events.map((ev, i) => (
        <div key={i} className="text-sm border-b border-gray-700 py-1">
          <span className="font-bold text-yellow-400">[{ev.severity}]</span> {ev.event_type} → {ev.message}
        </div>
      ))}
    </div>
  );
}
