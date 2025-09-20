import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../api/client";

export interface Trade {
  id: number;
  symbol: string;
  side: string;
  quantity: number;
  entry_price: number;
  exit_price?: number | null;
  pnl?: number | null;
  created_at: string;
}

export interface EquityPoint {
  trade_id: number;
  balance: number;
}

// --- API Calls ---
async function fetchTrades(): Promise<Trade[]> {
  const res = await api.get("/trading/list");
  return res.data;
}

async function fetchEquity(): Promise<{ equity_curve: EquityPoint[] }> {
  const res = await api.get("/stats/equity");
  return res.data;
}

async function createTrade(payload: {
  symbol: string;
  side: string;
  quantity: number;
  entry_price: number;
}): Promise<Trade> {
  const res = await api.post("/trading/trade", payload);
  return res;
}

// --- React Query Hooks ---
export function useTrades() {
  return useQuery<Trade[]>({
    queryKey: ["trades"],
    queryFn: fetchTrades,
  });
}

export function useEquity() {
  return useQuery<{ equity_curve: EquityPoint[] }>({
    queryKey: ["equity"],
    queryFn: fetchEquity,
  });
}

export function useCreateTrade() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createTrade,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["trades"] });
      queryClient.invalidateQueries({ queryKey: ["equity"] });
    },
  });
}
