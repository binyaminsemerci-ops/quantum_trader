// Application entrypoint (fresh start baseline)
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Robust mount with simple fallback logging.
function mount() {
  const rootEl = document.getElementById('root');
  if (!rootEl) {
    console.error('[entry] Could not find #root element');
    return;
  }
  const root = ReactDOM.createRoot(rootEl);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  // eslint-disable-next-line no-console
  console.log('[entry] React application mounted');
}

mount();

export {}; // keep as a module
