import React, { useState, useEffect } from 'react';
import axios from 'axios';
import moment from 'moment';
import { Table, Badge, Button, Spinner } from 'react-bootstrap';

type Trade = {
  trade_id: number | string;
  symbol: string;
  side: 'BUY' | 'SELL' | string;
  quantity?: number;
  price?: number | null;
  timestamp?: string;
};

const TradeHistory: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteInProgress, setDeleteInProgress] = useState<number | string | null>(null);

  useEffect(() => {
    fetchTrades();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchTrades = async (): Promise<void> => {
    setLoading(true);
    try {
      const response = await axios.get<Trade[]>('/trades');
      const data = response.data ?? [];
      setTrades(Array.isArray(data) ? data : []);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching trades:', err);
      setError('Failed to load trade history. Please try again later.');
      setLoading(false);
    }
  };

  const handleDelete = async (tradeId: number | string): Promise<void> => {
    if (!window.confirm(`Delete trade ${tradeId}?`)) return;
    setDeleteInProgress(tradeId);
    try {
      await axios.delete<void>(`/trade/${tradeId}`);
      setTrades(prev => prev.filter(t => String(t.trade_id) !== String(tradeId)));
    } catch (err) {
      console.error('Error deleting trade:', err);
      alert('Failed to delete trade. Please try again.');
    } finally {
      setDeleteInProgress(null);
    }
  };

  if (loading) return <div className="trade-history loading">Loading trade history...</div>;
  if (error) return <div className="trade-history error">{error}</div>;

  return (
    <div className="trade-history">
      <h3>Trade History</h3>
      {trades.length === 0 ? (
        <p>No trades found. Create your first trade!</p>
      ) : (
        <Table striped bordered hover responsive>
          <thead>
            <tr>
              <th>ID</th>
              <th>Symbol</th>
              <th>Side</th>
              <th>Quantity</th>
              <th>Price</th>
              <th>Timestamp</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {trades.map(trade => (
              <tr key={String(trade.trade_id)}>
                <td>{trade.trade_id}</td>
                <td>{trade.symbol}</td>
                <td>
                  <Badge bg={trade.side === 'BUY' ? 'success' : 'danger'}>
                    {trade.side}
                  </Badge>
                </td>
                <td>{trade.quantity ?? '—'}</td>
                <td>{trade.price ?? 'Market'}</td>
                <td>{trade.timestamp ? moment(trade.timestamp).format('DD.MM.YYYY, HH:mm:ss') : '—'}</td>
                <td>
                  <Button 
                    variant="outline-danger" 
                    size="sm"
                    disabled={deleteInProgress !== null && String(deleteInProgress) === String(trade.trade_id)}
                    onClick={() => handleDelete(trade.trade_id)}
                  >
                    {deleteInProgress !== null && String(deleteInProgress) === String(trade.trade_id) ? (
                      <><Spinner size="sm" animation="border" /> Deleting...</>
                    ) : (
                      'Delete'
                    )}
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
      <Button variant="primary" onClick={fetchTrades}>Refresh</Button>
    </div>
  );
};

export default TradeHistory;

