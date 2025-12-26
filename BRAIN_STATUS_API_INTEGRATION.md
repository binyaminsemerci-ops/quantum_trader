# Brain Status API Integration Example

## Endpoint
```
GET https://api.quantumfond.com/brains/state
```

## Response Structure
```json
{
  "CEO_Brain": {
    "mode": "OPTIMIZE",
    "confidence": 0.92,
    "transition_log": [
      {"timestamp": 1766709977.73, "from": "OPTIMIZE", "to": "EXPAND"},
      {"timestamp": 1766711777.73, "from": "EXPAND", "to": "STABILIZE"}
    ]
  },
  "Strategy_Brain": {
    "mode": "STABILIZE",
    "confidence": 0.81,
    "transition_log": [...]
  },
  "Risk_Brain": {
    "mode": "EXPAND",
    "confidence": 0.87,
    "transition_log": [...]
  },
  "system_timestamp": 1766713577.73
}
```

## Operational Modes
- **OPTIMIZE**: Fine-tuning existing positions for maximum efficiency
- **EXPAND**: Actively seeking new opportunities and opening positions
- **STABILIZE**: Maintaining current positions, risk-averse stance
- **EMERGENCY**: Crisis mode, defensive posture, reducing exposure

## Frontend Integration (React/TypeScript)

### Basic Fetch
```typescript
import { useEffect, useState } from 'react';

interface BrainState {
  mode: string;
  confidence: number;
  transition_log: Array<{
    timestamp: number;
    from: string;
    to: string;
  }>;
}

interface BrainStatus {
  CEO_Brain: BrainState;
  Strategy_Brain: BrainState;
  Risk_Brain: BrainState;
  system_timestamp: number;
}

export function BrainDashboard() {
  const [brainData, setBrainData] = useState<BrainStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBrainStatus = async () => {
      try {
        const res = await fetch('https://api.quantumfond.com/brains/state');
        const data = await res.json();
        setBrainData(data);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch brain status:', err);
        setLoading(false);
      }
    };

    // Initial fetch
    fetchBrainStatus();

    // Refresh every 10 seconds
    const interval = setInterval(fetchBrainStatus, 10000);

    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading brain status...</div>;
  if (!brainData) return <div>Failed to load brain data</div>;

  return (
    <div className="brain-dashboard">
      <h2>🧠 Brain Status Monitor</h2>
      
      {/* CEO Brain */}
      <BrainCard 
        title="CEO Brain"
        brain={brainData.CEO_Brain}
        icon="👔"
      />
      
      {/* Strategy Brain */}
      <BrainCard 
        title="Strategy Brain"
        brain={brainData.Strategy_Brain}
        icon="🎯"
      />
      
      {/* Risk Brain */}
      <BrainCard 
        title="Risk Brain"
        brain={brainData.Risk_Brain}
        icon="🛡️"
      />
    </div>
  );
}

function BrainCard({ title, brain, icon }: { 
  title: string; 
  brain: BrainState; 
  icon: string;
}) {
  // Color coding based on mode
  const modeColor = {
    OPTIMIZE: 'text-blue-400',
    EXPAND: 'text-green-400',
    STABILIZE: 'text-yellow-400',
    EMERGENCY: 'text-red-400'
  }[brain.mode] || 'text-gray-400';

  // Confidence bar color
  const confidenceColor = brain.confidence > 0.8 
    ? 'bg-green-500' 
    : brain.confidence > 0.6 
      ? 'bg-yellow-500' 
      : 'bg-red-500';

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-2xl">
      <h3 className="text-xl font-semibold text-white mb-4">
        {icon} {title}
      </h3>
      
      {/* Current Mode */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400">Mode:</span>
        <span className={`font-bold ${modeColor}`}>
          {brain.mode}
        </span>
      </div>
      
      {/* Confidence Level */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <span className="text-gray-400">Confidence:</span>
          <span className="text-white font-bold">
            {(brain.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div 
            className={`${confidenceColor} h-2 rounded-full transition-all duration-300`}
            style={{ width: `${brain.confidence * 100}%` }}
          />
        </div>
      </div>
      
      {/* Recent Transitions */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <span className="text-gray-400 text-sm">Recent Transitions:</span>
        {brain.transition_log.slice(-3).reverse().map((transition, idx) => (
          <div key={idx} className="text-xs text-gray-500 mt-1">
            {new Date(transition.timestamp * 1000).toLocaleTimeString()}: 
            {' '}{transition.from} → {transition.to}
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Visualization Ideas

### System Mood Indicator
```typescript
function calculateSystemMood(brainData: BrainStatus): number {
  const avgConfidence = (
    brainData.CEO_Brain.confidence +
    brainData.Strategy_Brain.confidence +
    brainData.Risk_Brain.confidence
  ) / 3;
  return avgConfidence;
}

// Usage:
const systemMood = calculateSystemMood(brainData);
// Display as: "System Confidence: 87.3%"
```

### Mode Transition Timeline
```typescript
function getAllTransitions(brainData: BrainStatus) {
  const allTransitions = [
    ...brainData.CEO_Brain.transition_log.map(t => ({ ...t, brain: 'CEO' })),
    ...brainData.Strategy_Brain.transition_log.map(t => ({ ...t, brain: 'Strategy' })),
    ...brainData.Risk_Brain.transition_log.map(t => ({ ...t, brain: 'Risk' }))
  ].sort((a, b) => b.timestamp - a.timestamp).slice(0, 10);
  
  return allTransitions;
}

// Render as vertical timeline
```

### Aggregate Brain Consensus
```typescript
function getBrainConsensus(brainData: BrainStatus): {
  mode: string;
  agreement: number;
} {
  const modes = [
    brainData.CEO_Brain.mode,
    brainData.Strategy_Brain.mode,
    brainData.Risk_Brain.mode
  ];
  
  const modeCounts = modes.reduce((acc, mode) => {
    acc[mode] = (acc[mode] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  
  const dominantMode = Object.entries(modeCounts)
    .sort(([,a], [,b]) => b - a)[0];
  
  return {
    mode: dominantMode[0],
    agreement: dominantMode[1] / 3
  };
}

// Display: "🎯 Brain Consensus: STABILIZE (67% agreement)"
```

## API Validation

✅ Endpoint available: https://api.quantumfond.com/brains/state  
✅ CORS enabled for https://app.quantumfond.com  
✅ Returns 3 brain states with transition logs  
✅ Valid JSON structure  
✅ Timestamp included for synchronization  

## Next Steps (Future Phases)

1. **Historical Data**: Store brain state changes in database
2. **Real-time Updates**: WebSocket streaming for instant mode changes
3. **Alerting**: Notify on EMERGENCY mode or low confidence
4. **Inter-brain Correlation**: Analyze how brain decisions influence each other
5. **Performance Tracking**: Correlate brain modes with portfolio performance

---

>>> [Phase 7 Complete – Brains & Strategy module operational on api.quantumfond.com]
