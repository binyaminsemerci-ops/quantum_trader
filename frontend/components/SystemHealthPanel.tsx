// System Health Panel - displays microservices status
import type { DashboardSystemHealth } from '@/lib/types';
import { getServiceStatusColor, formatRelativeTime } from '@/lib/utils';
import DashboardCard from './DashboardCard';

interface SystemHealthPanelProps {
  system?: DashboardSystemHealth;
}

export default function SystemHealthPanel({ system }: SystemHealthPanelProps) {
  if (!system) {
    return (
      <DashboardCard title="Systemstatus">
        <div className="text-center text-gray-500 py-8">Loading system health...</div>
      </DashboardCard>
    );
  }

  return (
    <DashboardCard 
      title="Systemstatus"
      rightSlot={
        <span className={`badge ${getServiceStatusColor(system.overall_status)}`}>
          {system.overall_status}
        </span>
      }
      fullHeight
    >
      {/* Alerts summary */}
      {system.alerts_count > 0 && (
        <div className="mb-4 p-3 bg-warning/10 border border-warning/30 rounded text-sm">
          <div className="flex items-center justify-between">
            <span className="font-medium text-warning-dark">
              {system.alerts_count} aktiv{system.alerts_count > 1 ? 'e' : ''} varsl{system.alerts_count > 1 ? 'er' : ''}
            </span>
            {system.last_alert && (
              <span className="text-xs text-gray-600">
                {formatRelativeTime(system.last_alert)}
              </span>
            )}
          </div>
        </div>
      )}
      
      {/* Services list */}
      <div className="space-y-2">
        {system.services.map((service, idx) => (
          <div
            key={`${service.name}-${idx}`}
            className="flex items-center justify-between p-3 rounded bg-gray-50 dark:bg-slate-700/50"
          >
            <div className="flex-1">
              <div className="font-medium">{service.name}</div>
              {service.last_check && (
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  Last check: {formatRelativeTime(service.last_check)}
                </div>
              )}
            </div>
            
            <div className="flex items-center gap-3">
              {service.latency_ms !== undefined && (
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  {service.latency_ms}ms
                </span>
              )}
              <span className={`badge ${getServiceStatusColor(service.status)}`}>
                {service.status}
              </span>
            </div>
          </div>
        ))}
        
        {system.services.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            Ingen tjeneste-data
          </div>
        )}
      </div>
    </DashboardCard>
  );
}
