const API_BASE = 'http://localhost:8000';

export async function fetchTrades(): Promise<any> {
  const res = await fetch(`${API_BASE}/trades`);
  return res.json();
}

export async function fetchStats(): Promise<any> {
  const res = await fetch(`${API_BASE}/stats`);
  return res.json();
}

export async function fetchChart(): Promise<any> {
  const res = await fetch(`${API_BASE}/chart`);
  return res.json();
}

export async function fetchSettings(): Promise<any> {
  const res = await fetch(`${API_BASE}/settings`);
  return res.json();
}

export async function fetchBinance(): Promise<any> {
  const res = await fetch(`${API_BASE}/binance`);
  return res.json();
}
