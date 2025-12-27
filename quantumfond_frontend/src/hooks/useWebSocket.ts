import { useEffect, useState } from 'react';

/**
 * WebSocket hook for real-time data streaming via Redis pub/sub
 * @param channel - Redis channel name (e.g., "signals", "positions")
 */
export function useWebSocket(channel: string) {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    // Mock WebSocket connection (replace with actual WebSocket in production)
    // For now, poll the API every 2 seconds to simulate real-time updates
    const interval = setInterval(async () => {
      try {
        const apiUrl = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8026';
        
        if (channel === 'signals') {
          const res = await fetch(`${apiUrl}/ai/predict`);
          const json = await res.json();
          setData(json);
        } else if (channel === 'positions') {
          // Fetch last position (mock for now)
          setData({
            symbol: 'BTCUSDT',
            direction: 'BUY',
            price: 42000,
            tp: 42500,
            sl: 41500,
            trailing: 250,
            confidence: 0.85,
            model: 'PatchTST'
          });
        }
      } catch (error) {
        console.error(`WebSocket error for channel ${channel}:`, error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [channel]);

  return data;
}
