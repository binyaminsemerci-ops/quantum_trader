import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { DashboardProvider } from './hooks/useDashboardData';

const rootEl = document.getElementById('root');
if (rootEl) {
  ReactDOM.createRoot(rootEl).render(
    <React.StrictMode>
      <DashboardProvider>
        <App />
      </DashboardProvider>
    </React.StrictMode>
  );
}

export {};
