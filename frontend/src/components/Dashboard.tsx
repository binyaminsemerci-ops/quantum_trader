import React, { useEffect, useState } from 'react'
import axios from 'axios'

interface Trade {
  id: number
  symbol: string
  action: string
  quantity: number
  price: number
}

function Dashboard() {
  const [trades, setTrades] = useState<Trade[]>([])

  useEffect(() => {
    axios.get<Trade[]>("/api/trading/trades")
      .then(res => setTrades(res.data))
      .catch(err => console.error(err))
  }, [])

  return (
    <div>
      <h2>📊 Trades</h2>
      <ul>
        {trades.map(t => (
          <li key={t.id}>
            {t.symbol} - {t.action} - {t.quantity} @ {t.price}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default Dashboard
