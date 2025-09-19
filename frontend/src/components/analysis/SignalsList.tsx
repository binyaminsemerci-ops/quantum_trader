import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Table, Badge, Button, Spinner } from 'react-bootstrap';
import moment from 'moment';
import { Signal } from '../../types';

export default function SignalsList(): JSX.Element {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [running, setRunning] = useState<Record<string, boolean>>({});

  const fetchSignals = async () => {
    setLoading(true);
    try {
  const res = await axios.get<Signal[]>('/signals');
      const data = res?.data ?? [];
      setSignals(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Error fetching signals:', err);
      setSignals([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchSignals(); }, []);

  const doPost = async (path: string, key?: string) => {
    if (key) setRunning(prev => ({ ...prev, [key]: true }));
    try {
  await axios.post<void>(path);
      await fetchSignals();
    } catch (e) {
      console.error('post error', e);
    } finally {
      if (key) setRunning(prev => ({ ...prev, [key]: false }));
    }
  };

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-2">
        <h5 className="m-0">AI Signals</h5>
        <Button size="sm" onClick={fetchSignals} disabled={loading}>{loading ? <Spinner size="sm" /> : 'Refresh'}</Button>
      </div>

      <Table striped bordered hover size="sm">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Signal</th>
            <th>Confidence</th>
            <th>Time</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {signals.length === 0 && (
            <tr><td colSpan={5} className="text-center">{loading ? 'Loading...' : 'No signals'}</td></tr>
          )}
          {signals.map((s, idx) => (
            <tr key={String(s.id ?? s._id ?? idx)}>
              <td>{s.symbol ?? s.sym ?? '-'}</td>
              <td><Badge bg="secondary">{s.signal ?? s.signal_type ?? '-'}</Badge></td>
              <td>{(typeof s.confidence === 'number' ? `${Math.round(s.confidence * 100)}%` : (typeof s.confidence_score === 'number' ? `${Math.round(s.confidence_score * 100)}%` : '-'))}</td>
              <td>{s.timestamp ? moment(s.timestamp).format('YYYY-MM-DD HH:mm:ss') : '-'}</td>
              <td>
                <Button size="sm" className="me-2" onClick={() => doPost(`/predict/${encodeURIComponent(String(s.symbol ?? s.sym ?? ''))}`, String(s.symbol ?? s.sym ?? ''))} disabled={!!running[String(s.symbol ?? s.sym ?? '')]}>
                  {running[String(s.symbol ?? s.sym ?? '')] ? <Spinner size="sm" animation="border" /> : 'Generate'}
                </Button>
                <Button size="sm" variant="success" onClick={() => doPost(`/trade/signal/${encodeURIComponent(String(s.symbol ?? s.sym ?? ''))}`, String(s.symbol ?? s.sym ?? ''))} disabled={!!running[String(s.symbol ?? s.sym ?? '')]}>
                  {running[String(s.symbol ?? s.sym ?? '')] ? <Spinner size="sm" animation="border" /> : 'Execute'}
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
}
