import { useState, useEffect, useRef } from 'react';

interface WebSocketHookOptions {
  url: string;
  enabled?: boolean;
  reconnectInterval?: number;
  debounceMs?: number; // forsinker utsending av data state for å redusere re-renders
}

export function useWebSocket<T = any>(options: WebSocketHookOptions) {
  const { url, enabled = true, reconnectInterval = 3000, debounceMs = 0 } = options;
  const [data, setData] = useState<T | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const pendingDataRef = useRef<T | null>(null);

  const connect = () => {
    if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      setConnectionStatus('connecting');
      setError(null);
      
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus('connected');
        setError(null);
        console.log(`WebSocket connected: ${url}`);
      };

      ws.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          if (debounceMs > 0) {
            pendingDataRef.current = parsedData;
            if (!debounceTimerRef.current) {
              debounceTimerRef.current = setTimeout(() => {
                debounceTimerRef.current = null;
                if (pendingDataRef.current !== null) {
                  setData(pendingDataRef.current);
                  pendingDataRef.current = null;
                }
              }, debounceMs);
            }
          } else {
            setData(parsedData);
          }
        } catch (err) {
          console.warn('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        setConnectionStatus('disconnected');
        if (enabled && reconnectTimeoutRef.current === null) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = () => {
        setError('WebSocket connection error');
        setConnectionStatus('disconnected');
      };

    } catch (err: any) {
      setError(`Connection failed: ${err.message}`);
      setConnectionStatus('disconnected');
    }
  };

  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url, enabled, debounceMs]);

  const sendMessage = (message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  };

  return {
    data,
    connectionStatus,
    error,
    sendMessage
  };
}