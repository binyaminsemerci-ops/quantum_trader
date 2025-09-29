
import { useEffect, useState } from 'react';
import api, { setDefaultExchange } from '../utils/api';

type Message = { type: 'success' | 'error'; text: string } | null;
type HealthCapabilities = {
  exchanges?: Record<string, boolean>;
  live_market_data?: boolean;
};

const EXCHANGES = [
  { id: 'binance', label: 'Binance' },
  { id: 'coinbase', label: 'Coinbase' },
  { id: 'kucoin', label: 'KuCoin' },
];

export default function Settings(): JSX.Element {
  const [selectedExchange, setSelectedExchange] = useState<string>('binance');
  const [apiKey, setApiKey] = useState<string>('');
  const [apiSecret, setApiSecret] = useState<string>('');
  const [enableLiveData, setEnableLiveData] = useState<boolean>(false);
  const [message, setMessage] = useState<Message>(null);
  const [capabilities, setCapabilities] = useState<HealthCapabilities>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const base = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
        const res = await fetch(`${base}/health`);
        if (cancelled) return;
        if (res.ok) {
          const payload = await res.json();
          setCapabilities({
            exchanges: payload?.capabilities?.exchanges,
            live_market_data:
              payload?.capabilities?.live_market_data ?? payload?.secrets?.live_market_data,
          });
        }
      } catch (err) {
        console.warn('failed to fetch health summary', err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const saveSettings = async () => {
    setSaving(true);
    setMessage(null);
    try {
      const key = apiKey.trim();
      const secret = apiSecret.trim();
      if (!key || !secret) {
        setMessage({ type: 'error', text: 'Please provide API key and secret before saving.' });
        return;
      }

      const payload: Record<string, string | boolean> = {
        [`${selectedExchange.toUpperCase()}_API_KEY`]: key,
        [`${selectedExchange.toUpperCase()}_API_SECRET`]: secret,
        ENABLE_LIVE_MARKET_DATA: enableLiveData,
      };

      const response = await api.saveSettings(payload);
      if (response && 'error' in response) {
        throw new Error(response.error);
      }

      await setDefaultExchange(selectedExchange as any);
      setMessage({ type: 'success', text: 'Settings saved successfully.' });
      setApiSecret('');
    } catch (err: unknown) {
      const text = err instanceof Error ? err.message : String(err);
      setMessage({ type: 'error', text: `Failed to save settings: ${text}` });
    } finally {
      setSaving(false);
    }
  };

  const exchangeCapabilities = capabilities.exchanges ?? {};

  return (
    <div className="p-6 dark:bg-gray-900 dark:text-white min-h-screen">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Settings</h1>
        {loading ? (
          <span className="text-sm text-slate-400">Loading system info...</span>
        ) : (
          <span className="text-sm text-slate-400">Health info fetched</span>
        )}
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded shadow space-y-4">
        <div>
          <label htmlFor="exchange-select" className="block text-sm font-semibold">
            Default Exchange
          </label>
          <select
            id="exchange-select"
            value={selectedExchange}
            onChange={(e) => setSelectedExchange(e.target.value)}
            className="border rounded p-2 w-full text-black"
          >
            {EXCHANGES.map((ex) => (
              <option key={ex.id} value={ex.id}>
                {ex.label}
                {exchangeCapabilities && exchangeCapabilities[ex.id] ? ' (configured)' : ''}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="api-key" className="block text-sm font-semibold">
            {selectedExchange === 'binance'
              ? 'Binance API Key'
              : `${selectedExchange.charAt(0).toUpperCase()}${selectedExchange.slice(1)} API Key`}
          </label>
          <input
            id="api-key"
            type="text"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="border rounded p-2 w-full text-black"
            placeholder="Enter API key"
            autoComplete="off"
          />
        </div>

        <div>
          <label htmlFor="api-secret" className="block text-sm font-semibold">
            {selectedExchange === 'binance'
              ? 'Binance API Secret'
              : `${selectedExchange.charAt(0).toUpperCase()}${selectedExchange.slice(1)} API Secret`}
          </label>
          <input
            id="api-secret"
            type="password"
            value={apiSecret}
            onChange={(e) => setApiSecret(e.target.value)}
            className="border rounded p-2 w-full text-black"
            placeholder="Enter API secret"
            autoComplete="off"
          />
          <p className="text-xs text-slate-500 mt-1">Secrets are stored only in-memory for demo purposes.</p>
        </div>

        <label className="flex items-center gap-3 text-sm">
          <input
            type="checkbox"
            checked={enableLiveData}
            onChange={(e) => setEnableLiveData(e.target.checked)}
          />
          Enable live market data (requires ccxt + API credentials)
        </label>

        <button
          onClick={saveSettings}
          disabled={saving}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow disabled:opacity-60"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>

        {message && (
          <p className={`mt-2 text-sm ${message.type === 'success' ? 'text-emerald-600' : 'text-rose-500'}`}>
            {message.text}
          </p>
        )}
      </div>
    </div>
  );
}
