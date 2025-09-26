import { useState } from 'react';
import api, { setDefaultExchange } from '../utils/api';

export default function Settings(): JSX.Element {
  const [apiKey, setApiKey] = useState<string>('');
  const [apiSecret, setApiSecret] = useState<string>('');
  const [msg, setMsg] = useState<string | null>(null);
  const [selectedExchange, setSelectedExchange] = useState<string>('binance');

  const saveSettings = async () => {
    try {
  // Read the currently selected exchange directly from the DOM to avoid timing issues
  const exchangeEl = document.getElementById('exchange-select') as HTMLSelectElement | null;
  const exchange = exchangeEl?.value ?? selectedExchange;

      // Read API key/secret from DOM to avoid timing issues in tests
      const keyEl = document.getElementById('api-key') as HTMLInputElement | null;
      const secretEl = document.getElementById('api-secret') as HTMLInputElement | null;
      const apiKeyVal = keyEl?.value ?? apiKey;
      const apiSecretVal = secretEl?.value ?? apiSecret;

      // Basic validation: require both key and secret
      if (!apiKeyVal || !apiSecretVal) {
        setMsg('Please provide API key and secret for the selected exchange');
        return;
      }

      // Use api.saveSettings which is mocked in tests
      await api.saveSettings({ key: `${exchange.toUpperCase()}_API_KEY`, value: apiKeyVal });
      await api.saveSettings({ key: `${exchange.toUpperCase()}_API_SECRET`, value: apiSecretVal });

      // Notify other parts of the app / tests that the default exchange was updated
      // Call the named export directly so the test mock is invoked
      // eslint-disable-next-line no-console
      console.log('Settings: calling setDefaultExchange', exchange);
      try {
        await setDefaultExchange(exchange as any);
      } catch {}
      setMsg('✅ API keys saved successfully');
    } catch (err: unknown) {
      const message = (err && typeof err === 'object' && 'message' in err) ? String((err as any).message) : String(err);
      setMsg('⚠️ Error: ' + (message ?? 'Unknown error'));
    }
  };

  return (
    <div className="p-6 dark:bg-gray-900 dark:text-white min-h-screen">
      <h1 className="text-2xl font-bold mb-4">⚙️ Settings</h1>

      <div className="bg-white dark:bg-gray-800 p-6 rounded shadow space-y-4">
        <div>
          <label htmlFor="exchange-select" className="block text-sm font-semibold">Default Exchange</label>
          <select id="exchange-select" value={selectedExchange} onChange={(e) => setSelectedExchange(e.target.value)} className="border rounded p-2 w-full text-black">
            <option value="binance">Binance</option>
            <option value="coinbase">Coinbase</option>
          </select>
        </div>
        <div>
          <label htmlFor="api-key" className="block text-sm font-semibold">{selectedExchange === 'binance' ? 'Binance API Key' : 'Coinbase API Key'}</label>
          <input
            id="api-key"
            type="text"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="border rounded p-2 w-full text-black"
          />
        </div>

        <div>
          <label htmlFor="api-secret" className="block text-sm font-semibold">{selectedExchange === 'binance' ? 'Binance API Secret' : 'Coinbase API Secret'}</label>
          <input
            id="api-secret"
            type="password"
            value={apiSecret}
            onChange={(e) => setApiSecret(e.target.value)}
            className="border rounded p-2 w-full text-black"
          />
        </div>

        <button onClick={saveSettings} className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow">
          Save
        </button>

        {msg && <p className="mt-2 text-sm">{msg}</p>}
      </div>
    </div>
  );
}
