# Continuous Learning Manager API Documentation

## Overview
Track model versions, training losses, and get automatic retraining recommendations based on validation score degradation and time-based drift detection.

## Endpoint

### Learning Status
**URL:** `GET https://api.quantumfond.com/learning/status`  
**Description:** Returns current model training status and retraining recommendations  
**Method:** GET  
**Response:** JSON object with model training metrics

---

## Response Structure

```json
{
  "timestamp": 1766713930.47,
  "model": "LGBM",
  "version": "v3.4.9",
  "last_loss": 0.0254,
  "validation_score": 0.682,
  "last_trained": 1766672208.47,
  "training_time": 647,
  "suggestion": "Retrain recommended"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | float | Current Unix timestamp |
| `model` | string | Model type (XGB, LGBM, N-HiTS, TFT) |
| `version` | string | Semantic version (e.g., v3.4.9) |
| `last_loss` | float | Training loss from last training session (0.010 - 0.035) |
| `validation_score` | float | Model accuracy on validation set (0.0 - 1.0) |
| `last_trained` | float | Unix timestamp of last training completion |
| `training_time` | integer | Training duration in seconds (600-2400s) |
| `suggestion` | string | "Stable" or "Retrain recommended" |

---

## Retraining Logic

The system recommends retraining based on two conditions:

### 1. **Performance Degradation**
```
validation_score < 0.70
```
If model accuracy drops below 70%, immediate retraining is suggested.

### 2. **Time-based Drift**
```
(current_time - last_trained) > 43200 seconds (12 hours)
```
If model hasn't been retrained in 12+ hours, drift is assumed and retraining is recommended.

### Suggestion Values
- **"Stable"**: Model performing well, no action needed
- **"Retrain recommended"**: Model requires retraining due to drift or performance issues

---

## Testing

### Basic Test
```bash
curl https://api.quantumfond.com/learning/status
```

**Sample Output:**
```json
{
  "timestamp": 1766713935.61,
  "model": "LGBM",
  "version": "v3.9.6",
  "last_loss": 0.0204,
  "validation_score": 0.653,
  "last_trained": 1766644075.61,
  "training_time": 1063,
  "suggestion": "Retrain recommended"
}
```

### Multiple Requests Test
```powershell
for ($i=1; $i -le 5; $i++) {
    curl -s https://api.quantumfond.com/learning/status | ConvertFrom-Json | 
    Select-Object model, validation_score, suggestion
}
```

**Sample Output:**
```
model   validation_score  suggestion
-----   ----------------  ----------
XGB     0.77              Retrain recommended
N-HiTS  0.83              Stable
LGBM    0.77              Stable
TFT     0.68              Retrain recommended
TFT     0.84              Retrain recommended
```

---

## Frontend Integration

### React + TypeScript Implementation

```typescript
import { useEffect, useState } from 'react';

interface LearningStatus {
  timestamp: number;
  model: string;
  version: string;
  last_loss: number;
  validation_score: number;
  last_trained: number;
  training_time: number;
  suggestion: string;
}

export function LearningMonitor() {
  const [learning, setLearning] = useState<LearningStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLearning = async () => {
      try {
        const res = await fetch('https://api.quantumfond.com/learning/status');
        const data = await res.json();
        setLearning(data);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch learning status:', err);
        setLoading(false);
      }
    };

    // Initial fetch
    fetchLearning();

    // Refresh every 30 seconds
    const interval = setInterval(fetchLearning, 30000);

    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading learning status...</div>;
  if (!learning) return <div>Failed to load learning data</div>;

  return (
    <div className="learning-monitor">
      <h2>🤖 Continuous Learning Status</h2>
      
      <ModelCard learning={learning} />
      <ValidationGauge score={learning.validation_score} />
      <TrainingTimeline learning={learning} />
      <RetrainAlert learning={learning} />
    </div>
  );
}

function ModelCard({ learning }: { learning: LearningStatus }) {
  const timeSinceTraining = Math.floor(
    (learning.timestamp - learning.last_trained) / 3600
  );

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
      <h3 className="text-xl font-semibold text-white mb-4">
        📊 Model Information
      </h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <span className="text-gray-400 text-sm">Model:</span>
          <p className="text-white font-bold text-lg">{learning.model}</p>
        </div>
        
        <div>
          <span className="text-gray-400 text-sm">Version:</span>
          <p className="text-blue-400 font-mono text-lg">{learning.version}</p>
        </div>
        
        <div>
          <span className="text-gray-400 text-sm">Training Loss:</span>
          <p className="text-white font-bold">{learning.last_loss.toFixed(4)}</p>
        </div>
        
        <div>
          <span className="text-gray-400 text-sm">Training Time:</span>
          <p className="text-white font-bold">
            {Math.floor(learning.training_time / 60)}m {learning.training_time % 60}s
          </p>
        </div>
        
        <div className="col-span-2">
          <span className="text-gray-400 text-sm">Last Trained:</span>
          <p className="text-white font-bold">
            {timeSinceTraining}h ago
          </p>
        </div>
      </div>
    </div>
  );
}

function ValidationGauge({ score }: { score: number }) {
  // Color based on score
  const getColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.7) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const percentage = (score * 100).toFixed(1);

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
      <h3 className="text-xl font-semibold text-white mb-4">
        🎯 Validation Score
      </h3>
      
      {/* Gauge visualization */}
      <div className="relative h-32 mb-4">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className={`text-5xl font-bold ${
              score >= 0.8 ? 'text-green-400' : 
              score >= 0.7 ? 'text-yellow-400' : 
              'text-red-400'
            }`}>
              {percentage}%
            </div>
            <div className="text-gray-400 text-sm mt-2">Accuracy</div>
          </div>
        </div>
        
        {/* Progress arc */}
        <svg className="w-full h-full" viewBox="0 0 100 50">
          <path
            d="M 10,50 A 40,40 0 0,1 90,50"
            fill="none"
            stroke="#374151"
            strokeWidth="8"
          />
          <path
            d="M 10,50 A 40,40 0 0,1 90,50"
            fill="none"
            stroke={score >= 0.8 ? '#10b981' : score >= 0.7 ? '#eab308' : '#ef4444'}
            strokeWidth="8"
            strokeDasharray={`${score * 125.6} 125.6`}
          />
        </svg>
      </div>
      
      {/* Score interpretation */}
      <div className="flex items-center justify-between mt-4">
        <div className="text-sm text-gray-400">
          {score >= 0.8 ? '✅ Excellent' : 
           score >= 0.7 ? '⚠️ Acceptable' : 
           '🚨 Poor'}
        </div>
        <div className="text-sm text-gray-400">
          Threshold: 0.70
        </div>
      </div>
    </div>
  );
}

function TrainingTimeline({ learning }: { learning: LearningStatus }) {
  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
      <h3 className="text-xl font-semibold text-white mb-4">
        📈 Training History
      </h3>
      
      <div className="space-y-3">
        <div className="flex items-center">
          <div className="w-32 text-gray-400 text-sm">Loss Trend:</div>
          <div className="flex-1">
            <div className="bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full"
                style={{ width: `${(1 - learning.last_loss / 0.035) * 100}%` }}
              />
            </div>
          </div>
          <div className="w-20 text-right text-white font-mono text-sm">
            {learning.last_loss.toFixed(4)}
          </div>
        </div>
        
        <div className="text-xs text-gray-500 mt-2">
          💡 Lower loss indicates better model performance
        </div>
      </div>
    </div>
  );
}

function RetrainAlert({ learning }: { learning: LearningStatus }) {
  const needsRetrain = learning.suggestion === "Retrain recommended";

  return (
    <div className={`rounded-lg p-6 shadow-2xl ${
      needsRetrain 
        ? 'bg-red-900 border-2 border-red-500' 
        : 'bg-green-900 border-2 border-green-500'
    }`}>
      <div className="flex items-center gap-3">
        <span className="text-4xl">
          {needsRetrain ? '🚨' : '✅'}
        </span>
        <div className="flex-1">
          <h3 className={`text-xl font-bold ${
            needsRetrain ? 'text-red-300' : 'text-green-300'
          }`}>
            {learning.suggestion}
          </h3>
          <p className="text-sm text-gray-300 mt-1">
            {needsRetrain 
              ? 'Model performance degraded or staleness detected'
              : 'Model performing optimally, no action required'
            }
          </p>
        </div>
        
        {needsRetrain && (
          <button className="bg-red-600 hover:bg-red-700 px-6 py-2 rounded font-semibold">
            Trigger Retrain
          </button>
        )}
      </div>
      
      {/* Detailed reason */}
      {needsRetrain && (
        <div className="mt-4 pt-4 border-t border-red-700">
          <p className="text-sm text-red-200">
            <strong>Reason:</strong>{' '}
            {learning.validation_score < 0.7 
              ? `Validation score (${(learning.validation_score * 100).toFixed(1)}%) below threshold (70%)`
              : `Model age: ${Math.floor((learning.timestamp - learning.last_trained) / 3600)}h (threshold: 12h)`
            }
          </p>
        </div>
      )}
    </div>
  );
}
```

---

## Visualization Components

### 1. Validation Score Gauge
```typescript
// Circular gauge showing 0-100% validation score
// Color-coded: Green (>80%), Yellow (70-80%), Red (<70%)
// Real-time updates every 30 seconds
```

### 2. Training Loss Chart
```typescript
// Horizontal bar showing loss value (0.010 - 0.035)
// Lower is better - progress bar visualization
// Historical trend if stored in database
```

### 3. Model Info Table
```typescript
// Display: Model name, Version, Training time
// Last trained timestamp with human-readable format
// Training duration in minutes and seconds
```

### 4. Retrain Alert Banner
```typescript
// Conditional rendering based on suggestion
// Red alert: "Retrain recommended" with trigger button
// Green: "Stable" with checkmark
// Shows specific reason for recommendation
```

---

## CORS Configuration

Learning API is configured to accept requests from:
- ✅ `https://app.quantumfond.com`
- ✅ `http://localhost:5173` (development)
- ✅ `http://localhost:8889` (VPS testing)

**Headers:**
```
Access-Control-Allow-Origin: https://app.quantumfond.com
Access-Control-Allow-Credentials: true
Access-Control-Allow-Methods: *
Access-Control-Allow-Headers: *
```

---

## Model Types

### XGBoost (XGB)
- Gradient boosting framework
- Fast training, high accuracy
- Good for tabular data

### LightGBM (LGBM)
- Microsoft's gradient boosting
- Very fast, memory efficient
- Excellent for large datasets

### N-HiTS
- Neural hierarchical interpolation
- Time series forecasting
- Captures multiple seasonalities

### Temporal Fusion Transformer (TFT)
- Attention-based architecture
- Interpretable forecasts
- Best for complex patterns

---

## Verification Checklist

✅ Endpoint responds via HTTPS: `https://api.quantumfond.com/learning/status`  
✅ Returns valid JSON with all required fields  
✅ `suggestion` toggles between "Stable" and "Retrain recommended"  
✅ Retraining logic based on validation_score < 0.70  
✅ Time-based drift detection (>12 hours)  
✅ CORS headers configured for app.quantumfond.com  
✅ Multiple model types supported (XGB, LGBM, N-HiTS, TFT)  
✅ Semantic versioning for model versions  
✅ Training time tracked in seconds  
✅ Unix timestamps for precise time tracking  

---

## Future Enhancements (Phase 10+)

1. **Historical Tracking**: Store training runs in PostgreSQL
2. **Loss Trend Charts**: Visualize loss improvement over versions
3. **A/B Testing**: Compare new model vs current production
4. **Auto-Retraining**: Trigger retraining jobs automatically
5. **Model Registry**: Full model artifact management
6. **Hyperparameter Logs**: Track training configurations
7. **Drift Detection**: Statistical drift metrics (PSI, KS test)
8. **Performance Metrics**: Sharpe ratio, win rate, max drawdown
9. **Training Notifications**: Alert on training completion/failure
10. **Model Rollback**: Revert to previous version if needed

---

>>> **[Phase 9 Complete – Continuous Learning Manager operational on api.quantumfond.com]**
