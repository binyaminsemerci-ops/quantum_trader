import { useEffect, useMemo, useState } from 'react';
import api, { setDefaultExchange } from '../utils/api';
import Toast from '../components/Toast';

type Message = { type: 'success' | 'error' | 'info'; text: string } | null;
type HealthCapabilities = {
  exchanges?: Record<string, boolean>;
  live_market_data?: boolean;
};

type StoredSettings = Record<string, string | boolean | undefined>;

const EXCHANGES = [
  { id: 'binance', label: 'Binance' },
  { id: 'coinbase', label: 'Coinbase' },
  { id: 'kucoin', label: 'KuCoin' },
];

const mask = (value: string): string => {
  if (!value) return '';
  if (value.length <= 4) return '*'.repeat(value.length);
  return `${'*'.repeat(value.length - 4)}${value.slice(-4)}`;
};

export default function Settings(): JSX.Element {
  const [selectedExchange, setSelectedExchange] = useState<string>('binance');
  const [apiKey, setApiKey] = useState<string>('');
  const [apiSecret, setApiSecret] = useState<string>('');
  const [enableLiveData, setEnableLiveData] = useState<boolean>(false);
  const [toast, setToast] = useState<Message>(null);
  const [capabilities, setCapabilities] = useState<HealthCapabilities>({});
  const [storedSettings, setStoredSettings] = useState<StoredSettings>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadHealth() {
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
          setEnableLiveData(Boolean(payload?.capabilities?.live_market_data ?? payload?.secrets?.live_market_data));
          setHealthError(null);
        } else {
          setHealthError(`Health check failed: HTTP ${res.status}`);
        }
      } catch (err) {
        if (!cancelled) {
          setHealthError('Failed to reach backend health endpoint');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    async function loadSettings() {
      try {
        const response = await api.getSettings();
        if ('error' in response && response.error) {
          setToast({ type: 'info', text: 'No saved settings yet.' });
          return;
        }
        const data = (response as { data?: StoredSettings }).data;
        if (data) {
          setStoredSettings(data);
          const defaultExchange = data.DEFAULT_EXCHANGE;
          if (typeof defaultExchange === 'string' && EXCHANGES.some((ex) => ex.id === defaultExchange)) {
            setSelectedExchange(defaultExchange);
          }
          if (typeof data.ENABLE_LIVE_MARKET_DATA === 'boolean') {
            setEnableLiveData(Boolean(data.ENABLE_LIVE_MARKET_DATA));
          }
        }
      } catch (err) {
        // swallow for now; settings endpoint is optional in demo mode
      }
    }

    loadHealth();
    loadSettings();
    return () => {
      cancelled = true;
    };
  }, []);

  const saveSettings = async () => {
    setSaving(true);
    setToast(null);
    try {
      const key = apiKey.trim();
      const secret = apiSecret.trim();
      if (!key || !secret) {
        setToast({ type: 'error', text: 'Please provide API key and secret before saving.' });
        return;
      }

      const payload: Record<string, string | boolean> = {
        [`${selectedExchange.toUpperCase()}_API_KEY`]: key,
        [`${selectedExchange.toUpperCase()}_API_SECRET`]: secret,
        ENABLE_LIVE_MARKET_DATA: enableLiveData,
        DEFAULT_EXCHANGE: selectedExchange,
      };

      const response = await api.saveSettings(payload);
      if (response && 'error' in response && response.error) {
        throw new Error(response.error);
      }

      await setDefaultExchange(selectedExchange as any);
      setToast({ type: 'success', text: 'Settings saved successfully.' });
      setApiSecret('');
      setStoredSettings((prev) => ({ ...prev, ...payload }));
    } catch (err: unknown) {
      const text = err instanceof Error ? err.message : String(err);
      setToast({ type: 'error', text: `Failed to save settings: ${text}` });
    } finally {
      setSaving(false);
    }
  };

  const exchangeCapabilities = capabilities.exchanges ?? {};
  const liveStatus = useMemo(() => {
    if (loading) return 'Checking live-data capabilities…';
    if (healthError) return healthError;
    return enableLiveData ? 'Live market data is enabled.' : 'Live market data is disabled (demo mode).';
  }, [enableLiveData, healthError, loading]);

  const configuredKey = (exchangeId: string) => {
    const key = storedSettings[`${exchangeId.toUpperCase()}_API_KEY`];
    return typeof key === 'string' && key.length ? mask(key) : null;
  };

  const saveDisabled = saving;

  return (
    <div className="p-6 dark:bg-gray-900 dark:text-white min-h-screen space-y-4">
      <Toast message={toast?.text} type={toast?.type ?? 'info'} onClose={() => setToast(null)} />

      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Settings</h1>
        {loading ? (
          <span className="text-sm text-slate-400">Loading system info...</span>
        ) : (
          <span className="text-sm text-slate-400">{liveStatus}</span>
        )}
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded shadow space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
                  {exchangeCapabilities && exchangeCapabilities[ex.id] ? ' (available)' : ''}
                </option>
              ))}
            </select>
          </div>
          <div className="text-sm text-slate-500 flex flex-col justify-end">
            <span>Current key status: {configuredKey(selectedExchange) ? `Configured (${configuredKey(selectedExchange)})` : 'Not configured'}</span>
          </div>
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
          <p className="text-xs text-slate-500 mt-1">Secrets are stored in-memory for demo purposes only.</p>
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
          disabled={saveDisabled}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow disabled:opacity-60"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>

        <div className="text-xs text-slate-500 space-y-1">
          <div>Configured exchanges: {
            EXCHANGES.map((ex) => configuredKey(ex.id) ? ex.label : null).filter(Boolean).join(', ') || 'none'
          }</div>
          <div>Health check source: {healthError ? 'unreachable' : (enableLiveData ? 'live' : 'demo')}.</div>
        </div>
      </div>
    </div>
  );
}
