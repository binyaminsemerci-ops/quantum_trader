// WebSocket client for real-time dashboard updates
import type { DashboardEvent } from './types';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
const ADMIN_TOKEN = process.env.NEXT_PUBLIC_ADMIN_TOKEN || '';

export type EventHandler = (event: DashboardEvent) => void;

export class DashboardWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Set<EventHandler> = new Set();
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000;
  private pingInterval: NodeJS.Timeout | null = null;

  constructor(url: string = WS_URL) {
    this.url = `${url}/ws/dashboard?token=${ADMIN_TOKEN}`;
  }

  /**
   * Connect to dashboard WebSocket
   */
  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('[DashboardWS] Already connected');
      return;
    }

    try {
      console.log('[DashboardWS] Connecting to', this.url);
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('[DashboardWS] Connected');
        this.reconnectAttempts = 0;
        this.startPingInterval();
      };

      this.ws.onmessage = (event) => {
        try {
          const data: DashboardEvent = JSON.parse(event.data);
          console.log('[DashboardWS] Event received:', data.type);
          
          // Notify all handlers
          this.handlers.forEach(handler => {
            try {
              handler(data);
            } catch (error) {
              console.error('[DashboardWS] Handler error:', error);
            }
          });
        } catch (error) {
          console.error('[DashboardWS] Message parse error:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[DashboardWS] Error:', error);
      };

      this.ws.onclose = () => {
        console.log('[DashboardWS] Disconnected');
        this.stopPingInterval();
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('[DashboardWS] Connection failed:', error);
      this.attemptReconnect();
    }
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    console.log('[DashboardWS] Disconnecting');
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    this.stopPingInterval();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.reconnectAttempts = 0;
  }

  /**
   * Subscribe to events
   * 
   * @param handler - Callback function to handle events
   * @returns Unsubscribe function
   */
  subscribe(handler: EventHandler): () => void {
    this.handlers.add(handler);
    
    return () => {
      this.handlers.delete(handler);
    };
  }

  /**
   * Send ping to keep connection alive
   */
  private sendPing(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send('ping');
    }
  }

  /**
   * Start ping interval (every 30 seconds)
   */
  private startPingInterval(): void {
    this.pingInterval = setInterval(() => {
      this.sendPing();
    }, 30000); // 30 seconds
  }

  /**
   * Stop ping interval
   */
  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Attempt to reconnect with exponential backoff
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[DashboardWS] Max reconnect attempts reached');
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    console.log(`[DashboardWS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Get connection status
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
export const dashboardWebSocket = new DashboardWebSocket();

// Helper functions
export const connectDashboardWS = () => dashboardWebSocket.connect();
export const disconnectDashboardWS = () => dashboardWebSocket.disconnect();
export const subscribeToDashboardEvents = (handler: EventHandler) => 
  dashboardWebSocket.subscribe(handler);
