import { useEffect, useState } from 'react';
import { safePct } from '../utils/formatters';

export default function AI() {
  const [models, setModels] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/ai/models')
      .then(res => res.json())
      .then(data => setModels(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">AI Models</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {models?.models?.map((model: any, idx: number) => (
          <div key={idx} className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold text-white">{model.name}</h3>
                <p className="text-sm text-gray-400">v{model.version}</p>
              </div>
              <span className={`px-3 py-1 rounded text-xs font-semibold ${
                model.status === 'active' ? 'bg-green-900/30 text-green-400' : 'bg-gray-700 text-gray-400'
              }`}>
                {model.status}
              </span>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Accuracy</span>
                <span className="text-white font-semibold">
                  {safePct(model.accuracy, 1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Last Prediction</span>
                <span className="text-white text-sm">
                  {model.last_prediction 
                    ? new Date(model.last_prediction).toLocaleTimeString()
                    : 'N/A'
                  }
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
