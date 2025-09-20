import React from "react";

interface ErrorBannerProps {
  show?: boolean;
  message?: React.ReactNode;
  role?: string;
}

export default function ErrorBanner({ show = false, message = '', role = 'alert' }: ErrorBannerProps): JSX.Element | null {
  if (!show) return null;

  return (
    <div role={role} className="bg-red-600 text-white p-3 rounded-lg shadow mb-4">
      {message}
    </div>
  );
}
