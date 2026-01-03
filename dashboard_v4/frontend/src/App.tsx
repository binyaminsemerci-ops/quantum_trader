import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import Overview from './pages/Overview';
import AIEngine from './pages/AIEngine';
import Portfolio from './pages/Portfolio';
import Risk from './pages/Risk';
import SystemHealth from './pages/SystemHealth';
import RLIntelligence from './pages/RLIntelligence';
import Grafana from './pages/Grafana';

function Navigation() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Overview', icon: '🏠' },
    { path: '/ai', label: 'AI Engine', icon: '🤖' },
    { path: '/rl', label: 'RL Intelligence', icon: '🧠' },
    { path: '/portfolio', label: 'Portfolio', icon: '💼' },
    { path: '/risk', label: 'Risk', icon: '⚠️' },
    { path: '/system', label: 'System', icon: '⚙️' },
    { path: '/grafana', label: 'Grafana', icon: '📊' }
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
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-900 text-white">
        <Navigation />
        
        <main className="container mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/ai" element={<AIEngine />} />
            <Route path="/rl" element={<RLIntelligence />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/risk" element={<Risk />} />
            <Route path="/system" element={<SystemHealth />} />
            <Route path="/grafana" element={<Grafana />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
