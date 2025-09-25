import React, { useEffect, useState } from 'react';
import api, { setDefaultExchange } from '../utils/api';

type SettingsResponse = Record<string, string | undefined>;

export default function Settings(): JSX.Element {
  const [apiKey, setApiKey] = useState<string>('');
  const [apiSecret, setApiSecret] = useState<string>('');
  const [coinbaseKey, setCoinbaseKey] = useState<string>('');
  const [coinbaseSecret, setCoinbaseSecret] = useState<string>('');
  const [kucoinKey, setKucoinKey] = useState<string>('');
  const [kucoinSecret, setKucoinSecret] = useState<string>('');
  const [defaultExchange, setDefaultExchangeLocal] = useState<string>('binance');
  const [msg, setMsg] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [showSecrets, setShowSecrets] = useState(false);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await api.getSettings();
        const settings: SettingsResponse = res && (res as any).data ? (res as any).data : {};
        if (!mounted) return;
        setApiKey(settings.BINANCE_API_KEY || '');
        setApiSecret(settings.BINANCE_API_SECRET || '');
        setCoinbaseKey(settings.COINBASE_API_KEY || '');
        setCoinbaseSecret(settings.COINBASE_API_SECRET || '');
        setKucoinKey(settings.KUCOIN_API_KEY || '');
        setKucoinSecret(settings.KUCOIN_API_SECRET || '');
        const de = settings.DEFAULT_EXCHANGE || 'binance';
        setDefaultExchangeLocal(de);
        setDefaultExchange(de);
      } catch (err) {
        // ignore
      }
    })();
    return () => { mounted = false; };
  }, []);

  const saveSettings = async () => {
    // basic client-side validation: if defaultExchange is set to an exchange,
    // require its key/secret to be provided (at least non-empty) as a hint to user
    const missing: string[] = [];
    if (defaultExchange === 'binance' && (!apiKey || !apiSecret)) missing.push('Binance key/secret');
    if (defaultExchange === 'coinbase' && (!coinbaseKey || !coinbaseSecret)) missing.push('Coinbase key/secret');
    if (defaultExchange === 'kucoin' && (!kucoinKey || !kucoinSecret)) missing.push('KuCoin key/secret');
    if (missing.length) {
      setMsg('⚠️ Please provide: ' + missing.join(', '));
      return;
    }

    try {
      setSaving(true);
      const payload: Record<string, unknown> = {
        BINANCE_API_KEY: apiKey,
        BINANCE_API_SECRET: apiSecret,
        COINBASE_API_KEY: coinbaseKey,
        COINBASE_API_SECRET: coinbaseSecret,
        KUCOIN_API_KEY: kucoinKey,
        KUCOIN_API_SECRET: kucoinSecret,
        DEFAULT_EXCHANGE: defaultExchange,
      };
      const resp = await api.saveSettings(payload);
      if ((resp as any).error) {
        setMsg('❌ Failed to save API keys');
      } else {
        setMsg('✅ API keys saved successfully');
        setDefaultExchange((payload.DEFAULT_EXCHANGE as string) || 'binance');
        // show saved indicator briefly
        setTimeout(() => setMsg(null), 3000);
      }
    } catch (err: unknown) {
      const message = (err && typeof err === 'object' && 'message' in err) ? String((err as any).message) : String(err);
      setMsg('⚠️ Error: ' + (message ?? 'Unknown error'));
    }
    finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-6 dark:bg-gray-900 dark:text-white min-h-screen">
      <h1 className="text-2xl font-bold mb-4">⚙️ Settings</h1>

        <div className="bg-white dark:bg-gray-800 p-6 rounded shadow space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="relative">
            <label htmlFor="binance-key" className="block text-sm font-semibold">Binance API Key</label>
            <div className="flex">
              <input
                id="binance-key"
                type="text"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="border rounded p-2 w-full text-black"
              />
              <button
                type="button"
                className="ml-2 px-2"
                onClick={() => navigator.clipboard?.writeText(apiKey)}
                aria-label="Copy Binance API Key"
              >
                Copy
              </button>
            </div>
          </div>

            <div>
              <label htmlFor="binance-secret" className="block text-sm font-semibold">Binance API Secret</label>
              <div className="flex">
                <input
                  id="binance-secret"
                  type={showSecrets ? 'text' : 'password'}
                  value={apiSecret}
                  onChange={(e) => setApiSecret(e.target.value)}
                  className="border rounded p-2 w-full text-black"
                />
                <button type="button" className="ml-2 px-2" onClick={() => setShowSecrets((s) => !s)}>
                  {showSecrets ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>

          <div className="relative">
            <label htmlFor="coinbase-key" className="block text-sm font-semibold">Coinbase API Key</label>
            <div className="flex">
              <input
                id="coinbase-key"
                type="text"
                value={coinbaseKey}
                onChange={(e) => setCoinbaseKey(e.target.value)}
                className="border rounded p-2 w-full text-black"
              />
              <button type="button" className="ml-2 px-2" onClick={() => navigator.clipboard?.writeText(coinbaseKey)} aria-label="Copy Coinbase API Key">Copy</button>
            </div>
          </div>

            <div>
              <label htmlFor="coinbase-secret" className="block text-sm font-semibold">Coinbase API Secret</label>
              <div className="flex">
                <input
                  id="coinbase-secret"
                  type={showSecrets ? 'text' : 'password'}
                  value={coinbaseSecret}
                  onChange={(e) => setCoinbaseSecret(e.target.value)}
                  className="border rounded p-2 w-full text-black"
                />
                <button type="button" className="ml-2 px-2" onClick={() => setShowSecrets((s) => !s)}>
                  {showSecrets ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>

          <div className="relative">
            <label htmlFor="kucoin-key" className="block text-sm font-semibold">KuCoin API Key</label>
            <div className="flex">
              <input
                id="kucoin-key"
                type="text"
                value={kucoinKey}
                onChange={(e) => setKucoinKey(e.target.value)}
                className="border rounded p-2 w-full text-black"
              />
              <button type="button" className="ml-2 px-2" onClick={() => navigator.clipboard?.writeText(kucoinKey)} aria-label="Copy KuCoin API Key">Copy</button>
            </div>
          </div>

            <div>
              <label htmlFor="kucoin-secret" className="block text-sm font-semibold">KuCoin API Secret</label>
              <div className="flex">
                <input
                  id="kucoin-secret"
                  type={showSecrets ? 'text' : 'password'}
                  value={kucoinSecret}
                  onChange={(e) => setKucoinSecret(e.target.value)}
                  className="border rounded p-2 w-full text-black"
                />
                <button type="button" className="ml-2 px-2" onClick={() => setShowSecrets((s) => !s)}>
                  {showSecrets ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>
        </div>

        <div className="mt-4">
          <label className="block text-sm font-semibold">Default Exchange</label>
          <select value={defaultExchange} onChange={(e) => { setDefaultExchangeLocal(e.target.value); setDefaultExchange(e.target.value); }} className="border rounded p-2 w-full text-black">
            <option value="binance">Binance</option>
            <option value="coinbase">Coinbase</option>
            <option value="kucoin">KuCoin</option>
          </select>
        </div>

        <div className="flex items-center space-x-4">
          <button disabled={saving} onClick={saveSettings} className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow disabled:opacity-50">
            {saving ? 'Saving...' : 'Save'}
          </button>
          <label className="text-sm">
            <input type="checkbox" checked={showSecrets} onChange={(e) => setShowSecrets(e.target.checked)} /> Show secrets
          </label>
        </div>

        {msg && <p className="mt-2 text-sm">{msg}</p>}
      </div>
    </div>
  );
}
