// Restored safe copy from frontend/legacy/App.jsx.bak
// Saved as App.restored.jsx to avoid overwriting current App.tsx

import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Tabs, Tab, Button, Alert } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

// Import components
import PriceChart from './components/dashboard/PriceChart';
import SentimentPanel from './components/dashboard/SentimentPanel';
import TradeHistory from './components/dashboard/TradeHistory';
import TradeForm from './components/trading/TradeForm';
import SignalsList from './components/analysis/SignalsList';

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [selectedInterval, setSelectedInterval] = useState('1h');
  const [systemStatus, setSystemStatus] = useState({ 
    backend: false, 
    checking: false,
    lastChecked: null
  });

  const checkBackendStatus = async () => {
    setSystemStatus(prev => ({ ...prev, checking: true }));
    try {
      const response = await fetch('http://localhost:8000');
      let data;
      try {
        data = await response.json();
      } catch {
        data = undefined;
      }
      setSystemStatus({
        backend: true,
        checking: false,
        lastChecked: new Date(),
        message: data?.message
      });
    } catch (error) {
      setSystemStatus({
        backend: false,
        checking: false,
        lastChecked: new Date(),
        error: error?.message ?? String(error)
      });
    }
  };

  useEffect(() => {
    checkBackendStatus();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <Container fluid>
          <Row>
            <Col>
              <h1>🚀 Quantum Trader</h1>
            </Col>
            <Col xs="auto" className="d-flex align-items-center">
              <div className="system-status me-3">
                <span className="status-label">Backend:</span>
                <span className={`status-indicator ${systemStatus.backend ? 'online' : 'offline'}`}>
                  {systemStatus.backend ? 'Online' : 'Offline'}
                </span>
              </div>
              <Button 
                variant="outline-light" 
                size="sm"
                onClick={checkBackendStatus}
                disabled={systemStatus.checking}
              >
                {systemStatus.checking ? 'Checking...' : 'Check Status'}
              </Button>
            </Col>
          </Row>
        </Container>
      </header>

      <main>
        <Container fluid className="mt-4">
          {!systemStatus.backend && (
            <Alert variant="danger">
              Backend API is not available. Please start the backend server.
            </Alert>
          )}
          
          <Row className="mb-4">
            <Col md={8}>
              <div className="symbol-selector mb-3">
                <h3>Market Data</h3>
                <div className="d-flex gap-2">
                  <div className="btn-group">
                    <Button 
                      variant={selectedSymbol === 'BTCUSDT' ? 'primary' : 'outline-primary'}
                      onClick={() => setSelectedSymbol('BTCUSDT')}
                    >
                      BTC
                    </Button>
                    <Button 
                      variant={selectedSymbol === 'ETHUSDT' ? 'primary' : 'outline-primary'}
                      onClick={() => setSelectedSymbol('ETHUSDT')}
                    >
                      ETH
                    </Button>
                    <Button 
                      variant={selectedSymbol === 'XRPUSDT' ? 'primary' : 'outline-primary'}
                      onClick={() => setSelectedSymbol('XRPUSDT')}
                    >
                      XRP
                    </Button>
                  </div>
                  
                  <div className="btn-group">
                    <Button 
                      variant={selectedInterval === '15m' ? 'primary' : 'outline-primary'}
                      onClick={() => setSelectedInterval('15m')}
                    >
                      15m
                    </Button>
                    <Button 
                      variant={selectedInterval === '1h' ? 'primary' : 'outline-primary'}
                      onClick={() => setSelectedInterval('1h')}
                    >
                      1h
                    </Button>
                    <Button 
                      variant={selectedInterval === '1d' ? 'primary' : 'outline-primary'}
                      onClick={() => setSelectedInterval('1d')}
                    >
                      1d
                    </Button>
                  </div>
                </div>
              </div>
              
              <div className="chart-wrapper">
                <PriceChart symbol={selectedSymbol} interval={selectedInterval} />
              </div>
            </Col>
            
            <Col md={4}>
              <TradeForm onTradeSubmitted={() => {}} />
            </Col>
          </Row>
          
          <Row className="mb-4">
            <Col>
              <Tabs defaultActiveKey="signals" className="mb-3">
                <Tab eventKey="signals" title="AI Signals">
                  <SignalsList />
                </Tab>
                <Tab eventKey="trades" title="Trade History">
                  <TradeHistory />
                </Tab>
                <Tab eventKey="sentiment" title="Market Sentiment">
                  <SentimentPanel symbol={selectedSymbol.substring(0, 3)} />
                </Tab>
              </Tabs>
            </Col>
          </Row>
        </Container>
      </main>
      
      <footer className="App-footer">
        <Container fluid>
          <p>Quantum Trader © 2025 - AI-Powered Cryptocurrency Trading Platform</p>
        </Container>
      </footer>
    </div>
  );
}

export default App;
