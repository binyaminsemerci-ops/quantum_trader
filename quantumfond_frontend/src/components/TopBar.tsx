interface TopBarProps {
  system?: {
    mode?: string;
    health?: string;
  };
}

export default function TopBar({ system }: TopBarProps) {
  const getHealthColor = (health?: string) => {
    switch (health) {
      case 'OK':
        return 'text-green-400';
      case 'WARNING':
        return 'text-yellow-400';
      case 'ERROR':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getModeColor = (mode?: string) => {
    return mode === 'LIVE' ? 'text-green-400' : 'text-yellow-400';
  };

  return (
    <header className="flex justify-between items-center px-6 py-4 border-b border-gray-800 bg-gray-900/50 backdrop-blur">
      <div className="flex items-center gap-4">
        <div className="font-bold text-xl text-green-400">
          QuantumFond OS
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-500">Mode:</span>
          <span className={`font-semibold ${getModeColor(system?.mode)}`}>
            {system?.mode || 'UNKNOWN'}
          </span>
        </div>
      </div>
      
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-500">Health:</span>
          <span className={`font-semibold ${getHealthColor(system?.health)}`}>
            {system?.health || 'UNKNOWN'}
          </span>
          <div className={`w-2 h-2 rounded-full ${
            system?.health === 'OK' ? 'bg-green-400' : 'bg-gray-400'
          } animate-pulse`}></div>
        </div>
        
        <div className="text-sm text-gray-500">
          {new Date().toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          })}
        </div>
        
        <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors">
          Logout
        </button>
      </div>
    </header>
  );
}
