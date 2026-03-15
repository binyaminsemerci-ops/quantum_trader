import { useEffect, useState } from 'react';

const API_BASE_URL = '/api';

interface StreamInfo {
  key: string;
  length: number;
}

interface StreamEntry {
  id: string;
  stream: string;
  symbol?: string;
  [key: string]: any;
}

export default function Replay() {
  const [streams, setStreams] = useState<Record<string, StreamInfo>>({});
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [selectedStream, setSelectedStream] = useState('');
  const [chain, setChain] = useState<StreamEntry[]>([]);
  const [streamEntries, setStreamEntries] = useState<StreamEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<'chain' | 'stream'>('chain');

  useEffect(() => {
    const init = async () => {
      try {
        const [streamsRes, symbolsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/replay/streams`),
          fetch(`${API_BASE_URL}/replay/symbols`),
        ]);
        if (streamsRes.ok) setStreams(await streamsRes.json());
        if (symbolsRes.ok) {
          const data = await symbolsRes.json();
          setSymbols(data.symbols || []);
        }
      } catch (err) {
        console.error('Failed to load replay data:', err);
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  const loadTradeChain = async (symbol: string) => {
    setSelectedSymbol(symbol);
    setMode('chain');
    try {
      const res = await fetch(`${API_BASE_URL}/replay/trade/${symbol}`);
      if (res.ok) {
        const data = await res.json();
        setChain(data.chain || []);
      }
    } catch (err) {
      console.error('Failed:', err);
    }
  };

  const loadStream = async (streamName: string) => {
    setSelectedStream(streamName);
    setMode('stream');
    try {
      const res = await fetch(`${API_BASE_URL}/replay/stream/${streamName}?count=100`);
      if (res.ok) {
        const data = await res.json();
        setStreamEntries(data.entries || []);
      }
    } catch (err) {
      console.error('Failed:', err);
    }
  };

  const STREAM_COLORS: Record<string, string> = {
    'trade.intent': 'text-blue-400 border-blue-600',
    'apply.plan': 'text-green-400 border-green-600',
    'apply.result': 'text-emerald-400 border-emerald-600',
    'exit.intent': 'text-orange-400 border-orange-600',
    'harvest.intent': 'text-yellow-400 border-yellow-600',
    'trade.closed': 'text-red-400 border-red-600',
  };

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading replay...</div></div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-green-400">Trade Replay</h1>
        <div className="flex gap-2">
          <button onClick={() => setMode('chain')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition ${mode === 'chain' ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>
            Decision Chain
          </button>
          <button onClick={() => setMode('stream')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition ${mode === 'stream' ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>
            Raw Streams
          </button>
        </div>
      </div>

      {/* Stream overview */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {Object.entries(streams).map(([name, info]) => (
          <div key={name}
            onClick={() => loadStream(name)}
            className={`p-3 bg-gray-800 rounded-xl border cursor-pointer hover:border-green-500 transition ${selectedStream === name && mode === 'stream' ? 'border-green-500' : 'border-gray-700'}`}>
            <div className={`text-sm font-medium ${STREAM_COLORS[name]?.split(' ')[0] || 'text-gray-300'}`}>{name}</div>
            <div className="text-2xl font-bold text-white">{info.length?.toLocaleString() ?? 0}</div>
            <div className="text-xs text-gray-500">entries</div>
          </div>
        ))}
      </div>

      {/* Decision chain mode */}
      {mode === 'chain' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <span className="text-gray-400 text-sm">Select symbol:</span>
            <div className="flex flex-wrap gap-2">
              {symbols.map((s) => (
                <button key={s} onClick={() => loadTradeChain(s)}
                  className={`px-3 py-1 rounded text-sm font-medium transition ${selectedSymbol === s ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          {selectedSymbol && chain.length === 0 && (
            <div className="bg-gray-800 rounded-xl border border-gray-700 p-8 text-center text-gray-500">
              No events found for {selectedSymbol} in recent streams
            </div>
          )}

          {chain.length > 0 && (
            <div className="relative">
              {/* Timeline line */}
              <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-700"></div>
              <div className="space-y-3">
                {chain.map((event, i) => {
                  const colorClass = STREAM_COLORS[event.stream] || 'text-gray-400 border-gray-600';
                  const [textColor, borderColor] = colorClass.split(' ');
                  return (
                    <div key={`${event.id}-${i}`} className="flex items-start gap-4 pl-2">
                      <div className={`w-3 h-3 rounded-full border-2 mt-1.5 flex-shrink-0 ${borderColor} bg-gray-900 z-10`}></div>
                      <div className={`flex-1 p-3 bg-gray-800 rounded-lg border border-gray-700`}>
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`text-xs font-medium ${textColor}`}>{event.stream}</span>
                          <span className="text-xs text-gray-600">{event.id}</span>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-1 text-sm">
                          {Object.entries(event)
                            .filter(([k]) => !['id', 'stream'].includes(k))
                            .map(([k, v]) => (
                              <div key={k}>
                                <span className="text-gray-500">{k}:</span>{' '}
                                <span className="text-gray-200">{String(v).substring(0, 60)}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Raw stream mode */}
      {mode === 'stream' && selectedStream && (
        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
            <h3 className="font-medium text-white">{selectedStream} — Recent entries</h3>
            <span className="text-sm text-gray-400">{streamEntries.length} entries</span>
          </div>
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-800">
                <tr className="border-b border-gray-700">
                  <th className="px-3 py-2 text-left text-gray-400 text-xs">ID</th>
                  <th className="px-3 py-2 text-left text-gray-400 text-xs">Symbol</th>
                  <th className="px-3 py-2 text-left text-gray-400 text-xs">Details</th>
                </tr>
              </thead>
              <tbody>
                {streamEntries.map((e, i) => (
                  <tr key={`${e.id}-${i}`} className="border-b border-gray-700/30 hover:bg-gray-750">
                    <td className="px-3 py-2 text-gray-500 font-mono text-xs">{e.id}</td>
                    <td className="px-3 py-2 text-white">{e.symbol || '-'}</td>
                    <td className="px-3 py-2 text-gray-300 text-xs">
                      {Object.entries(e)
                        .filter(([k]) => !['id', 'symbol'].includes(k))
                        .map(([k, v]) => `${k}=${String(v).substring(0, 40)}`)
                        .join(' | ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
