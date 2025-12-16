// TP Table - main table displaying TP performance across strategies/symbols
import type { TPDashboardKey } from '@/lib/tpDashboardTypes';

interface TpTableProps {
  entities: TPDashboardKey[];
  loading: boolean;
  onRowClick: (key: TPDashboardKey) => void;
}

export default function TpTable({ entities, loading, onRowClick }: TpTableProps) {
  if (loading) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-700 p-8">
        <div className="animate-pulse space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-12 bg-gray-200 dark:bg-slate-700 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (entities.length === 0) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-700 p-8 text-center">
        <p className="text-gray-500 dark:text-gray-400">
          No TP configurations found. Try adjusting your filters.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-700 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-slate-900 border-b border-gray-200 dark:border-slate-700">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Strategy
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Symbol
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-slate-700">
            {entities.map((entity, index) => (
              <tr
                key={`${entity.strategy_id}-${entity.symbol}`}
                onClick={() => onRowClick(entity)}
                className="hover:bg-gray-50 dark:hover:bg-slate-700 cursor-pointer transition-colors"
              >
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                  {entity.strategy_id}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                  {entity.symbol}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRowClick(entity);
                    }}
                    className="text-primary hover:text-primary/80 font-medium"
                  >
                    View Details
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="px-6 py-3 bg-gray-50 dark:bg-slate-900 border-t border-gray-200 dark:border-slate-700">
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Showing {entities.length} configuration{entities.length !== 1 ? 's' : ''}
        </p>
      </div>
    </div>
  );
}
