import { useState } from 'react';
import { api } from '../utils/api';
import type { ApiResponse } from '../utils/api';

export default function ApiTest(): JSX.Element {
  const [result, setResult] = useState<string | null>(null);

  async function onPing() {
    try {
  const res: ApiResponse<Record<string, unknown>> = await api.get('/health');
  if (res.error) setResult(`Error: ${res.error}`);
  else setResult(JSON.stringify(res.data ?? {}));
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
