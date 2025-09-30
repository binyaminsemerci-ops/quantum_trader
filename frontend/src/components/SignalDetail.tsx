import type { Signal } from "../api/signals";

type Props = {
  signal: Signal | null;
  onClose: () => void;
};

export default function SignalDetail({ signal, onClose }: Props) {
  if (!signal) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 text-black dark:text-white rounded-lg p-4 w-11/12 max-w-md shadow-lg">
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-lg font-semibold">Signal {signal.id}</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                try {
                  window.dispatchEvent(
                    new CustomEvent('focus-signal', { detail: { id: signal.id, timestamp: signal.timestamp } })
                  );
                } catch (e) {
                  // ignore in non-browser environments
                }
                onClose();
              }}
              className="px-3 py-1 bg-blue-600 text-white rounded"
            >
              Jump to chart
            </button>
            <button onClick={onClose} className="px-2 py-1 bg-gray-200 rounded">
              Close
            </button>
          </div>
        </div>
        <dl className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <dt className="text-xs text-gray-500">Symbol</dt>
            <dd className="font-mono">{signal.symbol}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Direction</dt>
            <dd>{signal.direction}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Score</dt>
            <dd>{Number.isFinite(signal.score) ? signal.score.toFixed(3) : '-'}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Confidence</dt>
            <dd>{signal.confidence != null ? `${Math.round(signal.confidence * 100)}%` : '-'}</dd>
          </div>
          <div className="col-span-2">
            <dt className="text-xs text-gray-500">Timestamp</dt>
            <dd>{signal.timestamp}</dd>
          </div>
          {signal.source && (
            <div>
              <dt className="text-xs text-gray-500">Source</dt>
              <dd className="text-xs text-slate-500">{signal.source}</dd>
            </div>
          )}
          {signal.details?.note && (
            <div className="col-span-2">
              <dt className="text-xs text-gray-500">Note</dt>
              <dd className="text-sm text-gray-600 dark:text-gray-300">{String(signal.details.note)}</dd>
            </div>
          )}
        </dl>
      </div>
    </div>
  );
}
