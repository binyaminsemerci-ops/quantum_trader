import { useState } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import Overview from './pages/Overview';
import AIEngine from './pages/AIEngine';
import Portfolio from './pages/Portfolio';
import Risk from './pages/Risk';
import SystemHealth from './pages/SystemHealth';
import RLIntelligence from './pages/RLIntelligence';
import Grafana from './pages/Grafana';
import Journal from './pages/Journal';
import Incidents from './pages/Incidents';
import Replay from './pages/Replay';
import Admin from './pages/Admin';
import Login from './pages/Login';

function Navigation({ username, onLogout }: { username: string | null; onLogout: () => void }) {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Overview', icon: '🏠' },
    { path: '/ai', label: 'AI Engine', icon: '🤖' },
    { path: '/rl', label: 'RL Intelligence', icon: '🧠' },
    { path: '/portfolio', label: 'Portfolio', icon: '💼' },
    { path: '/risk', label: 'Risk', icon: '⚠️' },
    { path: '/journal', label: 'Journal', icon: '📓' },
    { path: '/incidents', label: 'Incidents', icon: '🚨' },
    { path: '/replay', label: 'Replay', icon: '⏪' },
    { path: '/system', label: 'System', icon: '⚙️' },
    { path: '/admin', label: 'Admin', icon: '🔒' },
    { path: '/grafana', label: 'Grafana', icon: '📊' },
  ];

  return (
    <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-green-400">Quantum Hedge Fund</h1>
            <p className="text-sm text-gray-400">AI-Driven Trading System</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
              <span className="text-sm text-gray-400">Live</span>
            </div>
            {username ? (
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-400">{username}</span>
                <button onClick={onLogout} className="text-xs text-gray-500 hover:text-red-400 transition">Logout</button>
              </div>
            ) : (
              <Link to="/login" className="text-sm text-gray-400 hover:text-green-400 transition">Sign In</Link>
            )}
          </div>
        </div>
        
        <nav className="flex space-x-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`
                  px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                  ${isActive 
                    ? 'bg-green-500 text-white shadow-lg' 
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }
                `}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('qt_token'));
  const [role, setRole] = useState<string | null>(localStorage.getItem('qt_role'));
  const [username, setUsername] = useState<string | null>(localStorage.getItem('qt_user'));

  const handleLogin = (newToken: string, newRole: string, newUsername: string) => {
    setToken(newToken);
    setRole(newRole);
    setUsername(newUsername);
    localStorage.setItem('qt_token', newToken);
    localStorage.setItem('qt_role', newRole);
    localStorage.setItem('qt_user', newUsername);
  };

  const handleLogout = () => {
    setToken(null);
    setRole(null);
    setUsername(null);
    localStorage.removeItem('qt_token');
    localStorage.removeItem('qt_role');
    localStorage.removeItem('qt_user');
  };

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-900 text-white">
        <Navigation username={username} onLogout={handleLogout} />
        
        <main className="container mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/ai" element={<AIEngine />} />
            <Route path="/rl" element={<RLIntelligence />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/risk" element={<Risk />} />
            <Route path="/journal" element={<Journal token={token} />} />
            <Route path="/incidents" element={<Incidents token={token} />} />
            <Route path="/replay" element={<Replay />} />
            <Route path="/system" element={<SystemHealth />} />
            <Route path="/admin" element={<Admin token={token} role={role} />} />
            <Route path="/grafana" element={<Grafana />} />
            <Route path="/login" element={
              token ? <Navigate to="/" replace /> : <Login onLogin={handleLogin} />
            } />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
