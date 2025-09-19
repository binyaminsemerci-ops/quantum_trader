import React from "react";

type LoaderOverlayProps = {
  message?: React.ReactNode;
  show?: boolean;
};

/** Fullscreen loading overlay */
const LoaderOverlay: React.FC<LoaderOverlayProps> = ({ message = "Loading...", show = true }) => {
  if (!show) return null;
  return (
    <div className="fixed inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-lg flex flex-col items-center">
        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="mt-4 text-gray-700 font-medium">{message}</p>
      </div>
    </div>
  );
};

export default LoaderOverlay;
