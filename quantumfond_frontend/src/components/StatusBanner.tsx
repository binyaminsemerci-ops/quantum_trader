interface StatusBannerProps {
  type: 'success' | 'warning' | 'error' | 'info';
  message: string;
  onClose?: () => void;
}

export default function StatusBanner({ type, message, onClose }: StatusBannerProps) {
  const getStyles = () => {
    switch (type) {
      case 'success':
        return 'bg-green-900/30 border-green-700 text-green-400';
      case 'warning':
        return 'bg-yellow-900/30 border-yellow-700 text-yellow-400';
      case 'error':
        return 'bg-red-900/30 border-red-700 text-red-400';
      case 'info':
      default:
        return 'bg-blue-900/30 border-blue-700 text-blue-400';
    }
  };

  const getIcon = () => {
    switch (type) {
      case 'success':
        return '✓';
      case 'warning':
        return '⚠';
      case 'error':
        return '✕';
      case 'info':
      default:
        return 'ℹ';
    }
  };

  return (
    <div className={`flex items-center justify-between p-4 rounded-lg border ${getStyles()}`}>
      <div className="flex items-center gap-3">
        <span className="text-xl">{getIcon()}</span>
        <span className="font-medium">{message}</span>
      </div>
      
      {onClose && (
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ✕
        </button>
      )}
    </div>
  );
}
