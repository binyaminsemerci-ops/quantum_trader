import { useEffect, useState, useRef } from "react";

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: any) => void;
  onError?: (error: Event) => void;
  reconnectDelay?: number;
}

interface WebSocketState {
  data: any;
  isConnected: boolean;
  error: Event | null;
}

export function useWebSocket({
  url,
  onMessage,
  onError,
  reconnectDelay = 3000,
}: UseWebSocketOptions): WebSocketState {
  const [data, setData] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Event | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();

  useEffect(() => {
    let isMounted = true;

    const connect = () => {
      try {
        const ws = new WebSocket(url);

        ws.onopen = () => {
          if (isMounted) {
            console.log(`WebSocket connected: ${url}`);
            setIsConnected(true);
            setError(null);
          }
        };

        ws.onmessage = (event) => {
          if (isMounted) {
            try {
              const parsedData = JSON.parse(event.data);
              setData(parsedData);
              onMessage?.(parsedData);
            } catch (err) {
              console.error("Failed to parse WebSocket message:", err);
            }
          }
        };

        ws.onerror = (event) => {
          if (isMounted) {
            console.error("WebSocket error:", event);
            setError(event);
            setIsConnected(false);
            onError?.(event);
          }
        };

        ws.onclose = () => {
          if (isMounted) {
            console.log("WebSocket closed, reconnecting...");
            setIsConnected(false);
            wsRef.current = null;

            // Attempt to reconnect
            reconnectTimeoutRef.current = setTimeout(() => {
              if (isMounted) {
                connect();
              }
            }, reconnectDelay);
          }
        };

        wsRef.current = ws;
      } catch (err) {
        console.error("Failed to create WebSocket:", err);
        if (isMounted) {
          reconnectTimeoutRef.current = setTimeout(() => {
            if (isMounted) {
              connect();
            }
          }, reconnectDelay);
        }
      }
    };

    connect();

    return () => {
      isMounted = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [url, onMessage, onError, reconnectDelay]);

  return { data, isConnected, error };
}
