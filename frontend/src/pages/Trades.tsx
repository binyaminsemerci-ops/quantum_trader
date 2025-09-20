import React, { useEffect, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

interface Trade {
  id: number;
  pair: string;
  side: string;
  amount: number;
  price: number;
  timestamp: string;
}

export default function Trades() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [pair, setPair] = useState('');
  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [amount, setAmount] = useState<number>(0);
  const [price, setPrice] = useState<number>(0);

  async function fetchTrades() {
    const res = await fetch(`${API_BASE}/trades`);
    const data = await res.json();
    setTrades(data);
  }

  async function addTrade() {
    const res = await fetch(`${API_BASE}/trades`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pair, side, amount, price }),
    });
    if (res.ok) {
      await fetchTrades();
      setPair('');
      setAmount(0);
      setPrice(0);
    }
  }

  async function deleteTrade(id: number) {
    await fetch(`${API_BASE}/trades/${id}`, { method: 'DELETE' });
    await fetchTrades();
  }

  useEffect(() => {
    fetchTrades();
  }, []);

  return (
    <div>
      <h2>Trades</h2>

      {/* Ny trade form */}
      <div style={{ marginBottom: '1rem' }}>
        <input
          type="text"
          placeholder="Pair (f.eks. BTCUSDT)"
          value={pair}
          onChange={(e) => setPair(e.target.value)}
        />
        <select value={side} onChange={(e) => setSide(e.target.value as 'BUY' | 'SELL')}>
          <option value="BUY">BUY</option>
          <option value="SELL">SELL</option>
        </select>
        <input
          type="number"
          placeholder="Amount"
          value={amount}
          onChange={(e) => setAmount(Number(e.target.value))}
        />
        <input
          type="number"
          placeholder="Price"
          value={price}
          onChange={(e) => setPrice(Number(e.target.value))}
        />
        <button onClick={addTrade}>Legg til</button>
      </div>

      {/* Liste med trades */}
      <ul>
        {trades.map((t) => (
          <li key={t.id}>
            {t.pair} {t.side} {t.amount} @ {t.price} — {new Date(t.timestamp).toLocaleString()}
            <button onClick={() => deleteTrade(t.id)} style={{ marginLeft: '1rem' }}>
              Slett
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
