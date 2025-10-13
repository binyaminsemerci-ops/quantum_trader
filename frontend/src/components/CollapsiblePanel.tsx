import { useState } from 'react';
import type { ReactNode } from 'react';

interface CollapsiblePanelProps {
  title: string;
  icon?: string;
  children: ReactNode;
  defaultExpanded?: boolean;
  variant?: 'default' | 'compact' | 'minimal';
}

export default function CollapsiblePanel({ 
  title, 
  icon = '📊', 
  children, 
  defaultExpanded = true,
  variant = 'default'
}: CollapsiblePanelProps): JSX.Element {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const panelClasses = {
    default: 'bg-white dark:bg-gray-800 rounded-xl shadow-lg border dark:border-gray-700',
    compact: 'bg-white dark:bg-gray-800 rounded-lg shadow-md border dark:border-gray-700',
    minimal: 'bg-gray-50 dark:bg-gray-850 rounded-lg border dark:border-gray-600'
  };

  const headerClasses = {
    default: 'p-4 border-b dark:border-gray-700',
    compact: 'p-3 border-b dark:border-gray-700',
    minimal: 'p-2 border-b dark:border-gray-600'
  };

  const contentClasses = {
    default: 'p-4',
    compact: 'p-3',
    minimal: 'p-2'
  };

  return (
    <div className={`transition-all duration-200 ${panelClasses[variant]}`}>
      {/* Header med toggle */}
      <div 
        className={`flex items-center justify-between cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors ${headerClasses[variant]}`}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <h3 className={`font-semibold text-gray-800 dark:text-white ${variant === 'minimal' ? 'text-sm' : variant === 'compact' ? 'text-base' : 'text-lg'}`}>
            {title}
          </h3>
        </div>
        <div className={`transform transition-transform duration-200 ${isExpanded ? 'rotate-180' : 'rotate-0'}`}>
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Collapsible Content */}
      <div 
        className={`transition-all duration-200 overflow-hidden ${
          isExpanded ? 'max-h-screen opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className={contentClasses[variant]}>
          {children}
        </div>
      </div>
    </div>
  );
}