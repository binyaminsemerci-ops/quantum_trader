// frontend/src/pages/Settings.jsx
import { useState } from "react";

export default function Settings() {
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [msg, setMsg] = useState(null);

  const saveSettings = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: "BINANCE_API_KEY", value: apiKey }),
      });
      const res2 = await fetch("http://127.0.0.1:8000/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: "BINANCE_API_SECRET", value: apiSecret }),
      });

      if (res.ok && res2.ok) {
        setMsg("✅ API keys saved successfully");
      } else {
        setMsg("❌ Failed to save API keys");
      }
    } catch (err) {
      setMsg("⚠️ Error: " + err.message);
    }
  };

  return (
    <div className="p-6 dark:bg-gray-900 dark:text-white min-h-screen">
      <h1 className="text-2xl font-bold mb-4">⚙️ Settings</h1>

      <div className="bg-white dark:bg-gray-800 p-6 rounded shadow space-y-4">
        <div>
          <label className="block text-sm font-semibold">Binance API Key</label>
          <input
            type="text"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="border rounded p-2 w-full text-black"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold">
            Binance API Secret
          </label>
          <input
            type="password"
            value={apiSecret}
            onChange={(e) => setApiSecret(e.target.value)}
            className="border rounded p-2 w-full text-black"
          />
        </div>

        <button
          onClick={saveSettings}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow"
        >
          Save
        </button>

        {msg && <p className="mt-2 text-sm">{msg}</p>}
      </div>
    </div>
  );
}
