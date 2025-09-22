<<<<<<< Updated upstream
import { useState } from 'react';

export default function Sidebar(): JSX.Element {
  const [active, setActive] = useState('dashboard');

  return (
    <div className="w-64 h-screen bg-gray-900 text-white p-4 flex flex-col">
      <h2 className="text-2xl font-bold mb-8">Quantum Trader</h2>
      <nav className="flex-1">
        <ul>
          <li className={`mb-4 p-2 rounded-lg ${active === 'dashboard' ? 'bg-gray-700' : ''}`}>
            <a href="/dashboard" onClick={() => setActive('dashboard')}>Dashboard</a>
          </li>
          <li className={`mb-4 p-2 rounded-lg ${active === 'settings' ? 'bg-gray-700' : ''}`}>
            <a href="/settings" onClick={() => setActive('settings')}>Settings</a>
          </li>
        </ul>
      </nav>
      <div className="text-sm text-gray-400 mt-8">Quantum Trader © {new Date().getFullYear()}</div>
    </div>
  );
}
=======
// Auto-generated re-export stub
export { default } from './Sidebar.tsx';
>>>>>>> Stashed changes
