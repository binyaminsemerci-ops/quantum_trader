const API_BASE = "/api"; // proxes til FastAPI

async function request(endpoint, options = {}) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }
  return res.json();
}

export const api = {
  // Spot
  getSpotBalance: () => request("/binance/spot/balance"),
  getSpotPrice: (symbol) => request(`/binance/spot/price/${symbol}`),
  placeSpotOrder: (symbol, side, quantity) =>
    request("/binance/spot/order", {
      method: "POST",
      body: JSON.stringify({ symbol, side, quantity }),
    }),

  // Futures
  getFuturesBalance: () => request("/binance/futures/balance"),
  getFuturesPrice: (symbol) => request(`/binance/futures/price/${symbol}`),
  placeFuturesOrder: (symbol, side, quantity) =>
    request("/binance/futures/order", {
      method: "POST",
      body: JSON.stringify({ symbol, side, quantity }),
    }),
  getOpenFuturesOrders: (symbol) =>
    request(`/binance/futures/orders${symbol ? `?symbol=${symbol}` : ""}`),
  cancelFuturesOrder: (symbol, orderId) =>
    request(`/binance/futures/order/${symbol}/${orderId}`, { method: "DELETE" }),

  // Andre API-er
  getStats: () => request("/stats"),
  getTrades: () => request("/trades"),
  getChart: () => request("/chart"),
  getSettings: () => request("/settings"),
  saveSettings: (settings) =>
    request("/settings", { method: "POST", body: JSON.stringify(settings) }),
};
