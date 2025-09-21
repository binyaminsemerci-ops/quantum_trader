import { useEffect, useState } from 'react';

type ToastProps = {
  message?: string | JSX.Element;
  type?: 'info' | 'success' | 'error' | string;
  duration?: number;
  onClose?: () => void;
};

export default function Toast({ message, type = 'info', duration = 3000, onClose }: ToastProps): JSX.Element | null {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const id = setTimeout(() => {
      setVisible(false);
      onClose?.();
    }, duration);
    return () => clearTimeout(id);
  }, [duration, onClose]);

  if (!visible) return null;
  if (message === undefined || message === null) return null;

  const bg = type === 'success' ? 'bg-green-600' : type === 'error' ? 'bg-red-600' : 'bg-gray-700';

  return <div className={`fixed bottom-4 right-4 px-4 py-2 rounded shadow-lg text-white z-50 ${bg}`}>{message}</div>;
}
