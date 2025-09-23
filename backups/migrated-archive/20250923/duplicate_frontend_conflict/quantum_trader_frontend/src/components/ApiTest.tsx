<<<<<<< Updated upstream
import { useState } from 'react';
import { api } from '../utils/api';

export default function ApiTest(): JSX.Element {
  const [result, setResult] = useState<string | null>(null);

  async function onPing() {
    try {
      const res = await api.get('/health');
      setResult(JSON.stringify(res as any));
    } catch (e) {
      setResult('Error');
    }
  }

  return (
    <div className="p-4 bg-gray-800 text-white rounded">
      <button className="px-3 py-1 bg-blue-600 rounded" onClick={onPing}>Ping API</button>
      {result && <pre className="mt-2 text-xs">{result}</pre>}
    </div>
  );
}
=======
// Auto-generated re-export stub
export { default } from './ApiTest.tsx';
>>>>>>> Stashed changes
