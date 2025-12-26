export default function ControlPanel() {
  const handleAction = async (endpoint: string) => {
    try {
      await fetch(`https://api.quantumfond.com/control/${endpoint}`, { method: "POST" });
      alert(`Action sent: ${endpoint}`);
    } catch (err) {
      console.error(`Failed to send action: ${endpoint}`, err);
      alert(`Failed to send action: ${endpoint}`);
    }
  };

  return (
    <div className="bg-gray-900 p-4 rounded-xl shadow-md space-x-3">
      <button
        className="bg-green-600 px-3 py-1 rounded hover:bg-green-700 transition"
        onClick={() => handleAction("retrain")}
      >
        Retrain
      </button>
      <button
        className="bg-blue-600 px-3 py-1 rounded hover:bg-blue-700 transition"
        onClick={() => handleAction("heal")}
      >
        Heal
      </button>
      <button
        className="bg-red-600 px-3 py-1 rounded hover:bg-red-700 transition"
        onClick={() => handleAction("mode")}
      >
        Switch Mode
      </button>
    </div>
  );
}
