import { useMemo } from 'react';

/**
 * Hook to sanitize data objects and ensure numeric fields are safe
 * Prevents undefined/NaN/null from causing rendering errors
 */
export function useSafeData<T extends Record<string, any>>(data?: T | null): T {
  return useMemo(() => {
    if (!data || typeof data !== 'object') {
      return {} as T;
    }

    const cleaned: any = {};
    
    Object.entries(data).forEach(([key, val]) => {
      // Handle numbers
      if (typeof val === 'number') {
        cleaned[key] = isFinite(val) ? val : 0;
      }
      // Handle nested objects
      else if (val && typeof val === 'object' && !Array.isArray(val)) {
        cleaned[key] = useSafeData(val);
      }
      // Handle arrays
      else if (Array.isArray(val)) {
        cleaned[key] = val.map(item => 
          typeof item === 'object' ? useSafeData(item) : item
        );
      }
      // Handle other types with fallback
      else {
        cleaned[key] = val ?? (typeof val === 'number' ? 0 : '');
      }
    });

    return cleaned as T;
  }, [data]);
}
