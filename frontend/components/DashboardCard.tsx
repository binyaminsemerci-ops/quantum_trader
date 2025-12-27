/**
 * DashboardCard Component
 * Sprint 4 Del 3 + V3.0 Phase 10: Reusable card wrapper with enhanced styling
 * 
 * Provides consistent styling, layout, dark mode, hover effects, and optional title/actions
 */

import { ReactNode } from 'react';

interface DashboardCardProps {
  /** Panel title (shown in header) */
  title?: string;
  /** Optional element to show on the right side of header (e.g., badges, counts) */
  rightSlot?: ReactNode;
  /** Card content */
  children: ReactNode;
  /** Optional CSS class for custom styling */
  className?: string;
  /** Enable full height (flex-1) */
  fullHeight?: boolean;
  /** Disable hover effect */
  noHover?: boolean;
}

export default function DashboardCard({
  title,
  rightSlot,
  children,
  className = '',
  fullHeight = false,
  noHover = false
}: DashboardCardProps) {
  return (
    <div
      className={`
        bg-white dark:bg-slate-800 
        rounded-lg shadow-md 
        border border-gray-200 dark:border-slate-700
        transition-shadow duration-200
        ${noHover ? '' : 'hover:shadow-lg'}
        ${fullHeight ? 'h-full flex flex-col' : ''} 
        ${className}
      `}
    >
      {/* Header (if title or rightSlot provided) */}
      {(title || rightSlot) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-slate-700">
          {title && (
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              {title}
            </h2>
          )}
          {rightSlot && <div>{rightSlot}</div>}
        </div>
      )}
      
      {/* Content */}
      <div className={`p-4 ${fullHeight ? 'flex-1 overflow-auto' : ''}`}>
        {children}
      </div>
    </div>
  );
}
