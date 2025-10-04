import { useState, useCallback } from 'react';
import { useToast } from './useToast';

type AsyncOperationState = {
  isLoading: boolean;
  error: Error | null;
};

type UseAsyncOperationOptions = {
  successMessage?: string;
  errorMessage?: string;
  showSuccessToast?: boolean;
  showErrorToast?: boolean;
};

export function useAsyncOperation<T extends any[], R>(
  asyncFn: (...args: T) => Promise<R>,
  options: UseAsyncOperationOptions = {}
) {
  const [state, setState] = useState<AsyncOperationState>({
    isLoading: false,
    error: null,
  });
  
  const { showSuccess, showError } = useToast();
  
  const execute = useCallback(async (...args: T): Promise<R | null> => {
    setState({ isLoading: true, error: null });
    
    try {
      const result = await asyncFn(...args);
      
      if (options.showSuccessToast !== false && options.successMessage) {
        showSuccess(options.successMessage);
      }
      
      setState({ isLoading: false, error: null });
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('An unexpected error occurred');
      
      if (options.showErrorToast !== false) {
        const message = options.errorMessage || err.message || 'Operation failed';
        showError(message);
      }
      
      setState({ isLoading: false, error: err });
      return null;
    }
  }, [asyncFn, options, showSuccess, showError]);
  
  const reset = useCallback(() => {
    setState({ isLoading: false, error: null });
  }, []);
  
  return {
    ...state,
    execute,
    reset,
  };
}