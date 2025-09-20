import React, { useState, useEffect } from 'react';
import moment from 'moment';
import { Table, Badge, Button, Spinner } from 'react-bootstrap';
import type { Trade as SharedTrade } from '../../types';
import { api } from '../../utils/api';

const TradeHistory: React.FC = () => {
  const [trades, setTrades] = useState<SharedTrade[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteInProgress, setDeleteInProgress] = useState<number | string | null>(null);

  // fetchTrades is hoisted so UI elements (Refresh) can call it.
  const fetchTrades = async (): Promise<void> => {
    setLoading(true);
    let mounted = true;
    try {
      const resp = await api.getTrades();
      const data = resp?.data ?? [];
      if (!mounted) return;
      setTrades(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Error fetching trades:', err);
      setError('Failed to load trade history. Please try again later.');
    } finally {
      if (mounted) setLoading(false);
    }
  };

  useEffect(() => {
    let mounted = true;
    // call the shared fetch function, but guard mounted so async sets are safe
    (async () => {
      try {
        const resp = await api.getTrades();
        const data = resp?.data ?? [];
        if (!mounted) return;
        setTrades(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error('Error fetching trades:', err);
        if (mounted) setError('Failed to load trade history. Please try again later.');
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const handleDelete = async (tradeId: number | string): Promise<void> => {
    if (!window.confirm(`Delete trade ${tradeId}?`)) return;
    setDeleteInProgress(tradeId);
    try {
      const resp = await api.delete<void>(`/trade/${tradeId}`);
      if (resp?.error) {
        throw new Error(resp.error);
      }
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
                    onClick={() => { if (trade.trade_id != null) handleDelete(trade.trade_id); }}
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

