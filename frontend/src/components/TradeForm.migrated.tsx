import React, { useState, ChangeEvent, FormEvent } from 'react';
import '../styles/TradeForm.css';

type TradeFormData = {
  symbol: string;
  side: 'BUY' | 'SELL' | string;
  quantity: number | string;
  price: string | number;
};

type TradeFormProps = {
  onTradeSubmit?: (result: any) => void;
};

function TradeForm({ onTradeSubmit }: TradeFormProps) {
  const [formData, setFormData] = useState<TradeFormData>({
    symbol: 'BTCUSDT',
    side: 'BUY',
    quantity: 0.01,
    price: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target as HTMLInputElement;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'quantity' ? (value === '' ? '' : parseFloat(value)) : value
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/trade', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          symbol: formData.symbol,
          side: formData.side,
          quantity: formData.quantity,
          price: formData.price ? parseFloat(String(formData.price)) : null
        })
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data?.detail?.message || 'Failed to submit trade');
      }
      
      if (typeof onTradeSubmit === 'function') onTradeSubmit(data);
      
      if (formData.symbol === 'BTCUSDT') {
        setFormData((prev) => ({ ...prev, quantity: 0.01, price: '' }));
      }
    } catch (err: any) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="trade-form">
      <h2>New Trade</h2>
      {error && <div className="error">{error}</div>}
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="symbol">Symbol</label>
          <select
            id="symbol"
            name="symbol"
            value={formData.symbol}
            onChange={handleChange}
            required
          >
            <option value="BTCUSDT">BTC/USDT</option>
            <option value="ETHUSDT">ETH/USDT</option>
            <option value="XRPUSDT">XRP/USDT</option>
            <option value="ADAUSDT">ADA/USDT</option>
            <option value="DOGEUSDT">DOGE/USDT</option>
          </select>
        </div>
        
        <div className="form-group">
          <label>Side</label>
          <div className="radio-group">
            <label className="radio-label">
              <input
                type="radio"
                name="side"
                value="BUY"
                checked={formData.side === 'BUY'}
                onChange={handleChange}
              />
              Buy
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name="side"
                value="SELL"
                checked={formData.side === 'SELL'}
                onChange={handleChange}
              />
              Sell
            </label>
          </div>
        </div>
        
        <div className="form-group">
          <label htmlFor="quantity">Quantity</label>
          <input
            id="quantity"
            type="number"
            name="quantity"
            value={String(formData.quantity)}
            onChange={handleChange}
            step="0.001"
            min="0.001"
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="price">Price (optional, leave empty for market)</label>
          <input
            id="price"
            type="number"
            name="price"
            value={String(formData.price)}
            onChange={handleChange}
            step="0.01"
            min="0"
            placeholder="Market price"
          />
        </div>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Submitting...' : `${formData.side} ${formData.symbol}`}
        </button>
      </form>
    </div>
  );
}

export default TradeForm;
