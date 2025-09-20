// frontend/src/components/Toast.tsx
import React, { useEffect, useState } from "react";

type ToastProps = {
  message: React.ReactNode;
  type?: "info" | "success" | "error";
  duration?: number;
  onClose?: () => void;
};

const Toast: React.FC<ToastProps> = ({ message, type = "info", duration = 3000, onClose }) => {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const id = setTimeout(() => {
      setVisible(false);
      try {
        onClose?.();
      } catch (err) {
        // swallow errors from onClose to avoid breaking UI
        // eslint-disable-next-line no-console
        console.error('Toast onClose handler threw', err);
      }
    }, duration);
    return () => clearTimeout(id);
  }, [duration, onClose]);

  if (!visible) return null;

  return (
    <div
      className={`fixed bottom-4 right-4 px-4 py-2 rounded shadow-lg text-white z-50 ${
        type === "success" ? "bg-green-600" : type === "error" ? "bg-red-600" : "bg-gray-700"
      }`}
    >
      {message}
    </div>
  );
};

export default Toast;
