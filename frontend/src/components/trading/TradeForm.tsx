import React, { useState } from 'react';
import { Form, Button, InputGroup } from 'react-bootstrap';
import axios from 'axios';
import { Trade } from '../../types';

type TradeFormProps = {
  defaultSymbol?: string;
  onTradeExecuted?: (result: Trade) => void;
};

const TradeForm: React.FC<TradeFormProps> = ({ defaultSymbol = 'BTCUSDT', onTradeExecuted }) => {
  const [symbol, setSymbol] = useState<string>(defaultSymbol);
  const [amount, setAmount] = useState<number>(0.001);
  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [loading, setLoading] = useState<boolean>(false);

  const submitTrade = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol || typeof symbol !== 'string') return;
    setLoading(true);
    try {
      const res = await axios.post<Trade>('http://localhost:8000/trade', { symbol, amount, side });
      if (typeof onTradeExecuted === 'function') onTradeExecuted(res.data);
      // Minimal feedback for developer usage
      // eslint-disable-next-line no-alert
      alert('Trade submitted: ' + JSON.stringify(res.data));
    } catch (err) {
      console.error('Trade error', err);
      // eslint-disable-next-line no-alert
      alert('Failed to submit trade.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form onSubmit={submitTrade} className="trade-form">
      <Form.Group controlId="symbol">
        <Form.Label>Symbol</Form.Label>
        <Form.Control value={symbol} onChange={e => setSymbol(e.target.value)} />
      </Form.Group>

      <Form.Group controlId="amount">
        <Form.Label>Amount</Form.Label>
        <InputGroup>
          <Form.Control
            type="number"
            value={amount}
            onChange={e => setAmount(Number(e.target.value) || 0)}
            step="0.0001"
          />
        </InputGroup>
      </Form.Group>

      <Form.Group controlId="side">
        <Form.Label>Side</Form.Label>
        <Form.Select value={side} onChange={e => setSide((e.target.value as 'BUY' | 'SELL') ?? 'BUY')}>
          <option value="BUY">Buy</option>
          <option value="SELL">Sell</option>
        </Form.Select>
      </Form.Group>

      <Button type="submit" disabled={loading} className="mt-3">
        {loading ? 'Submitting...' : 'Submit Trade'}
      </Button>
    </Form>
  );
};

export default TradeForm;

