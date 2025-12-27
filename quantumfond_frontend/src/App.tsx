import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Overview from './pages/overview';
import Live from './pages/live';
import Risk from './pages/risk';
import AI from './pages/ai';
import Strategy from './pages/strategy';
import Performance from './pages/performance';
import System from './pages/system';
import Admin from './pages/admin';
import Incident from './pages/incident';
import Journal from './pages/journal';
import Replay from './pages/replay';
import { useState, useEffect } from 'react';

function App() {
  const [systemStatus, setSystemStatus] = useState({
    mode: 'LIVE',
    health: 'OK'
  });

  useEffect(() => {
    // Fetch system status
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(data => {
        setSystemStatus({
          mode: 'LIVE',
          health: data.status
        });
      })
      .catch(err => console.error('Failed to fetch system status:', err));
  }, []);

  return (
    <Router>
      <div className="flex min-h-screen bg-gray-950 text-gray-100">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <TopBar system={systemStatus} />
          <main className="flex-1 p-6">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/overview" element={<Overview />} />
              <Route path="/live" element={<Live />} />
              <Route path="/risk" element={<Risk />} />
              <Route path="/ai" element={<AI />} />
              <Route path="/strategy" element={<Strategy />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="/portfolio" element={<Performance />} />
              <Route path="/system" element={<System />} />
              <Route path="/admin" element={<Admin />} />
              <Route path="/incident" element={<Incident />} />
              <Route path="/journal" element={<Journal />} />
              <Route path="/replay" element={<Replay />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
