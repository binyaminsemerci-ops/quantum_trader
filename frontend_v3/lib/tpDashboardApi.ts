// API client for TP Dashboard endpoints
import type {
  TPDashboardKey,
  TPDashboardEntry,
  TPDashboardSummary,
} from './tpDashboardTypes';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class TPDashboardAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Fetch all TP entities (strategy_id + symbol combinations)
   */
  async fetchTpEntities(): Promise<TPDashboardKey[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/dashboard/tp/entities`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch TP entities: HTTP ${response.status}`);
      }

      const data = await response.json();
      return data as TPDashboardKey[];
    } catch (error) {
      console.error('[TPDashboardAPI] Error fetching entities:', error);
      throw error;
    }
  }

  /**
   * Fetch detailed entry for a specific strategy/symbol pair
   */
  async fetchTpEntry(strategyId: string, symbol: string): Promise<TPDashboardEntry> {
    try {
      const params = new URLSearchParams({
        strategy_id: strategyId,
        symbol: symbol,
      });

      const response = await fetch(
        `${this.baseUrl}/api/dashboard/tp/entry?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch TP entry: HTTP ${response.status}`);
      }

      const data = await response.json();
      return data as TPDashboardEntry;
    } catch (error) {
      console.error('[TPDashboardAPI] Error fetching entry:', error);
      throw error;
    }
  }

  /**
   * Fetch summary with best and worst performing configurations
   */
  async fetchTpSummary(limit: number = 10): Promise<TPDashboardSummary> {
    try {
      const params = new URLSearchParams({
        limit: limit.toString(),
      });

      const response = await fetch(
        `${this.baseUrl}/api/dashboard/tp/summary?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch TP summary: HTTP ${response.status}`);
      }

      const data = await response.json();
      
      // Backend returns {rows, total_pairs, generated_at}
      // Transform to expected format
      return {
        best: data.rows || [],
        worst: [],
      } as TPDashboardSummary;
    } catch (error) {
      console.error('[TPDashboardAPI] Error fetching summary:', error);
      throw error;
    }
  }
}

// Singleton instance
export const tpDashboardAPI = new TPDashboardAPI();

// Helper functions for easy import
export const fetchTpEntities = () => tpDashboardAPI.fetchTpEntities();
export const fetchTpEntry = (strategyId: string, symbol: string) =>
  tpDashboardAPI.fetchTpEntry(strategyId, symbol);
export const fetchTpSummary = (limit?: number) => tpDashboardAPI.fetchTpSummary(limit);
