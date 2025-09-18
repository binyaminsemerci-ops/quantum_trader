import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Table, Badge, Button, Spinner } from 'react-bootstrap';
import moment from 'moment';

type Signal = Record<string, any>;

const SignalsList: React.FC = () => {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [executingSignal, setExecutingSignal] = useState<string | null>(null);

  useEffect(() => { fetchSignals(); }, []);

  const fetchSignals = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/signals');
      setSignals(response.data ?? []);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching signals:', err);
      setError('Failed to load trading signals. Please try again later.');
      setLoading(false);
    }
  };

  const executeSignal = async (symbol: string) => {
    setExecutingSignal(symbol);
    try {
      const response = await axios.post(`/trade/signal/${symbol}`);
      alert(`Signal executed: ${response.data.execution?.message ?? 'OK'}`);
      fetchSignals();
    } catch (err) {
      console.error('Error executing signal:', err);
      alert('Failed to execute signal. Please try again.');
    } finally {
      setExecutingSignal(null);
    }
  };

  const generateSignal = async (symbol: string) => {
    setExecutingSignal(symbol);
    try {
      const response = await axios.get(`/predict/${symbol}`);
      alert(`Signal generated: ${response.data.signal} with confidence ${response.data.confidence}`);
      fetchSignals();
    } catch (err) {
      console.error('Error generating signal:', err);
      alert('Failed to generate signal. Please try again.');
    } finally {
      setExecutingSignal(null);
    }
  };

  if (loading) return <div className="signals-list loading">Loading signals...</div>;
  if (error) return <div className="signals-list error">{error}</div>;

  return (
    <div className="signals-list">
      <h3>AI Trading Signals</h3>
      <div className="generate-signals mb-4">
        <h4>Generate New Signal</h4>
        <div className="d-flex gap-2">
          <Button variant="outline-primary" onClick={() => generateSignal('BTCUSDT')} disabled={executingSignal === 'BTCUSDT'}>
            {executingSignal === 'BTCUSDT' ? <Spinner size="sm" animation="border" /> : null}
            BTC Signal
          </Button>
          <Button variant="outline-primary" onClick={() => generateSignal('ETHUSDT')} disabled={executingSignal === 'ETHUSDT'}>
            {executingSignal === 'ETHUSDT' ? <Spinner size="sm" animation="border" /> : null}
            ETH Signal
          </Button>
          <Button variant="outline-primary" onClick={() => generateSignal('XRPUSDT')} disabled={executingSignal === 'XRPUSDT'}>
            {executingSignal === 'XRPUSDT' ? <Spinner size="sm" animation="border" /> : null}
            XRP Signal
          </Button>
        </div>
      </div>

      {signals.length === 0 ? (
        <p>No signals found. Generate your first signal!</p>
      ) : (
        <Table striped bordered hover responsive>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Signal</th>
              <th>Confidence</th>
              <th>Timestamp</th>
              <th>Executed</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {signals.map(signal => {
              const rawSig = signal && (signal.signal ?? signal.signal_type ?? signal.signalType ?? signal.type);
              const sig = rawSig ? String(rawSig) : 'UNKNOWN';

              const rawConf = signal && (signal.confidence ?? signal.confidence_score ?? signal.confidencePercent ?? null);
              const confNumber = Number.isFinite(rawConf) ? rawConf : null;
              const confDisplay = confNumber === null ? '—' : `${Math.round(confNumber * 100)}%`;

              return (
                <tr key={signal.id}>
                  <td>{signal.symbol}</td>
                  <td>
                    <Badge bg={sig === 'BUY' ? 'success' : sig === 'SELL' ? 'danger' : 'secondary'}>{sig}</Badge>
                  </td>
                  <td>{confDisplay}</td>
                  <td>{moment(signal.timestamp).format('DD.MM.YYYY, HH:mm:ss')}</td>
                  <td><Badge bg={signal.executed ? 'success' : 'warning'}>{signal.executed ? 'Yes' : 'No'}</Badge></td>
                  <td>
                    {!signal.executed && sig !== 'HOLD' && (
                      <Button variant="success" size="sm" disabled={executingSignal === signal.symbol} onClick={() => executeSignal(signal.symbol)}>
                        {executingSignal === signal.symbol ? (<><Spinner size="sm" animation="border" /> Executing...</>) : ('Execute')}
                      </Button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      )}
      <Button variant="primary" onClick={fetchSignals}>Refresh</Button>
    </div>
  );
};

export default SignalsList;
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Table, Badge, Button, Spinner } from 'react-bootstrap';
import moment from 'moment';

type Signal = {
  id: string | number;
  symbol?: string;
  signal?: string | null;
  signal_type?: string | null;
  signalType?: string | null;
  type?: string | null;
  timestamp?: string;
  confidence?: number | null;
  executed?: boolean;
};

const SignalsList: React.FC = () => {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [executingSignal, setExecutingSignal] = useState<string | null>(null);

  useEffect(() => {
    fetchSignals();
  }, []);

  const fetchSignals = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/signals');
      setSignals(response.data ?? []);
      setLoading(false);
    } catch (err: any) {
      console.error('Error fetching signals:', err);
      setError('Failed to load trading signals. Please try again later.');
      setLoading(false);
    }
  };

  const executeSignal = async (symbol?: string) => {
    if (!symbol) return;
    setExecutingSignal(symbol);
    try {
      const response = await axios.post(`http://localhost:8000/trade/signal/${symbol}`);
      alert(`Signal executed: ${response.data.execution.message}`);
      fetchSignals();
    } catch (err) {
      console.error('Error executing signal:', err);
      alert('Failed to execute signal. Please try again.');
    } finally {
      setExecutingSignal(null);
    }
  };

  const generateSignal = async (symbol: string) => {
    setExecutingSignal(symbol);
    try {
      const response = await axios.get(`http://localhost:8000/predict/${symbol}`);
      alert(`Signal generated: ${response.data.signal} with confidence ${response.data.confidence}`);
      fetchSignals();
    } catch (err) {
      console.error('Error generating signal:', err);
      alert('Failed to generate signal. Please try again.');
    } finally {
      setExecutingSignal(null);
    }
  };

  if (loading) return <div className="signals-list loading">Loading signals...</div>;
  if (error) return <div className="signals-list error">{error}</div>;

  return (
    <div className="signals-list">
      <h3>AI Trading Signals</h3>

      <div className="generate-signals mb-4">
        <h4>Generate New Signal</h4>
        <div className="d-flex gap-2">
          <Button 
            variant="outline-primary" 
            onClick={() => generateSignal('BTCUSDT')}
            disabled={executingSignal === 'BTCUSDT'}
          >
            {executingSignal === 'BTCUSDT' ? <Spinner size="sm" animation="border" /> : null}
            BTC Signal
          </Button>
          <Button 
            variant="outline-primary" 
            onClick={() => generateSignal('ETHUSDT')}
            disabled={executingSignal === 'ETHUSDT'}
          >
            {executingSignal === 'ETHUSDT' ? <Spinner size="sm" animation="border" /> : null}
            ETH Signal
          </Button>
          <Button 
            variant="outline-primary" 
            onClick={() => generateSignal('XRPUSDT')}
            disabled={executingSignal === 'XRPUSDT'}
          >
            {executingSignal === 'XRPUSDT' ? <Spinner size="sm" animation="border" /> : null}
            XRP Signal
          </Button>
        </div>
      </div>

      {signals.length === 0 ? (
        <p>No signals found. Generate your first signal!</p>
      ) : (
        <Table striped bordered hover responsive>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Signal</th>
              <th>Confidence</th>
              <th>Timestamp</th>
              <th>Executed</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {signals.map(signal => {
              const rawSig = signal && (signal.signal ?? signal.signal_type ?? signal.signalType ?? signal.type);
              const sig = rawSig ? String(rawSig) : 'UNKNOWN';

              const rawConf = signal && (signal.confidence ?? (signal as any).confidence_score ?? null);
              const confNumber = Number.isFinite(rawConf as any) ? (rawConf as number) : null;
              const confDisplay = confNumber === null ? '—' : `${Math.round(confNumber * 100)}%`;

              return (
                <tr key={signal.id}>
                  <td>{signal.symbol}</td>
                  <td>
                    <Badge bg={
                      sig === 'BUY' ? 'success' : 
                      sig === 'SELL' ? 'danger' : 'secondary'
                    }>
                      {sig}
                    </Badge>
                  </td>
                  <td>{confDisplay}</td>
                  <td>{moment(signal.timestamp).format('DD.MM.YYYY, HH:mm:ss')}</td>
                  <td>
                    <Badge bg={signal.executed ? 'success' : 'warning'}>
                      {signal.executed ? 'Yes' : 'No'}
                    </Badge>
                  </td>
                  <td>
                    {!signal.executed && sig !== 'HOLD' && (
                      <Button 
                        variant="success" 
                        size="sm"
                        disabled={executingSignal === signal.symbol}
                        onClick={() => executeSignal(signal.symbol)}
                      >
                        {executingSignal === signal.symbol ? (
                          <><Spinner size="sm" animation="border" /> Executing...</>
                        ) : (
                          'Execute'
                        )}
                      </Button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      )}
      <Button variant="primary" onClick={fetchSignals}>Refresh</Button>
    </div>
  );
};

export default SignalsList;
