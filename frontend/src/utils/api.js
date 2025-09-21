
import axios from 'axios';
import { api as tsApi, trainModel as tsTrain, getPrediction as tsPredict } from './api.ts';

// Re-export the typed api object (thin JS wrapper). If the TS module isn't available
// at runtime (e.g. during some builds), fall back to minimal implementations.
export const api = tsApi || {
  get: (endpoint) => fetch(endpoint).then(r => r.json()),
  post: (endpoint, body) => fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json()),
  delete: (endpoint) => fetch(endpoint, { method: 'DELETE' }).then(r => r.json()),
};

// AI helpers (JS compatibility layer)
export const trainModel = tsTrain ? tsTrain : (symbol) => axios.post(`/api/ai/train/${symbol}`);
export const getPrediction = tsPredict ? tsPredict : (symbol) => axios.post(`/api/ai/predict/${symbol}`);

export default { trainModel, getPrediction, api };
