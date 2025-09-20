import React, { useState, useRef } from 'react';
import '../styles/TradeForm.css';
import type { Trade as SharedTrade } from '../types';
import { api } from '../utils/api';

type TradeFormProps = {
  onTradeSubmit?: (result: SharedTrade) => void;
};

type TradeFormState = {
  symbol: string;
  side: 'BUY' | 'SELL' | string;
  quantity: number | '';
  price: number | '';
};

const TradeForm: React.FC<TradeFormProps> = ({ onTradeSubmit }) => {
  const [formData, setFormData] = useState<TradeFormState>({ symbol: 'BTCUSDT', side: 'BUY', quantity: 0.01, price: '' });
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target as HTMLInputElement;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'quantity' || name === 'price' ? (value === '' ? '' : Number(value)) : value
    } as unknown as TradeFormState));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = { symbol: formData.symbol, side: formData.side, quantity: typeof formData.quantity === 'number' ? formData.quantity : Number(formData.quantity), price: formData.price === '' ? null : Number(formData.price) };
      const res = await api.post('/trade', payload);
      const data = res?.data as SharedTrade | undefined;
      if (res?.error) throw new Error(String(res.error));
      if (typeof onTradeSubmit === 'function' && data) onTradeSubmit(data);
      if (formData.symbol === 'BTCUSDT') setFormData(prev => ({ ...prev, quantity: 0.01, price: '' }));
    } catch (err: any) {
      setError(err?.message ?? String(err));
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  };

  return (
    <div className="trade-form">
      <h2>New Trade</h2>
      {error && <div className="error">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="symbol">Symbol</label>
          <select id="symbol" name="symbol" value={formData.symbol} onChange={handleChange} required>
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
            <label className="radio-label"><input type="radio" name="side" value="BUY" checked={formData.side === 'BUY'} onChange={handleChange} /> Buy</label>
            <label className="radio-label"><input type="radio" name="side" value="SELL" checked={formData.side === 'SELL'} onChange={handleChange} /> Sell</label>
          </div>
        </div>
        <div className="form-group">
          <label htmlFor="quantity">Quantity</label>
          <input id="quantity" type="number" name="quantity" value={formData.quantity as any} onChange={handleChange} step="0.001" min="0.001" required />
        </div>
        <div className="form-group">
          <label htmlFor="price">Price (optional)</label>
          <input id="price" type="number" name="price" value={formData.price as any} onChange={handleChange} step="0.01" min="0" placeholder="Market price" />
        </div>
        <button type="submit" disabled={loading}>{loading ? 'Submitting...' : `${formData.side} ${formData.symbol}`}</button>
      </form>
    </div>
  );
};

export default TradeForm;

