import { Link, useLocation } from 'react-router-dom';

interface SidebarProps {}

export default function Sidebar({}: SidebarProps) {
  const location = useLocation();
  
  const routes = [
    { path: '/overview', name: 'Overview', icon: '📊' },
    { path: '/live', name: 'Live Trades', icon: '⚡' },
    { path: '/risk', name: 'Risk', icon: '🛡️' },
    { path: '/ai', name: 'AI Models', icon: '🤖' },
    { path: '/strategy', name: 'Strategy', icon: '🎯' },
    { path: '/performance', name: 'Performance', icon: '📈' },
    { path: '/journal', name: 'Trade Journal', icon: '📖' },
    { path: '/replay', name: 'Trade Replay', icon: '🎬' },
    { path: '/system', name: 'System', icon: '⚙️' },
    { path: '/admin', name: 'Admin', icon: '👤' },
    { path: '/incident', name: 'Incidents', icon: '🚨' }
  ];

  return (
    <nav className="bg-gray-950 w-64 min-h-screen p-4 border-r border-gray-800">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-green-400 mb-2">QuantumFond</h1>
        <p className="text-xs text-gray-500">Hedge Fund OS</p>
      </div>
      
      <div className="space-y-1">
        {routes.map((route) => {
          const isActive = location.pathname === route.path;
          return (
            <Link
              key={route.path}
              to={route.path}
              className={`
                flex items-center gap-3 px-4 py-3 rounded-lg transition-all
                ${isActive 
                  ? 'bg-green-900/30 text-green-400 font-semibold' 
                  : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
                }
              `}
            >
              <span className="text-xl">{route.icon}</span>
              <span>{route.name}</span>
            </Link>
          );
        })}
      </div>
      
      <div className="absolute bottom-4 left-4 right-4">
        <div className="text-xs text-gray-600 text-center">
          v1.0.0 | {new Date().getFullYear()}
        </div>
      </div>
    </nav>
  );
}
