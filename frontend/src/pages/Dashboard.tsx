import React, { useEffect, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

export default function Dashboard() {
  const [trades, setTrades] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchTrades() {
      try {
        const res = await fetch(`${API_BASE}/trades`);
        const data = await res.json();
        setTrades(data);
      } catch (err) {
        console.error('Kunne ikke hente trades', err);
      } finally {
        setLoading(false);
      }
    }
    fetchTrades();
  }, []);

  return (
    <div>
      <h2>Dashboard</h2>
      {loading ? (
        <p>Laster trades...</p>
      ) : (
        <div>
          <p>Antall trades: {trades.length}</p>
          <ul>
            {trades.map((t) => (
              <li key={t.id}>
                {t.pair} {t.side} {t.amount} @ {t.price} (id {t.id})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
