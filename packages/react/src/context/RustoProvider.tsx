import {
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactElement,
} from 'react';
import type { InitializeConfig } from '@rusto/web';
import { RustoContext } from './RustoContext.js';
import { defaultOcrService } from '../services/ocrService.js';
import type {
  RustoContextValue,
  RustoProviderProps,
  RustoStatus,
} from '../types/index.js';

export function RustoProvider({
  children,
  config,
  autoInitialize = true,
}: RustoProviderProps): ReactElement {
  const [status, setStatus] = useState<RustoStatus>('idle');
  const [error, setError] = useState<Error | null>(null);

  const initialize = useCallback(
    async (overrideConfig?: InitializeConfig): Promise<void> => {
      try {
        setStatus('initializing');
        setError(null);
        await defaultOcrService.init(overrideConfig || config);
        setStatus('ready');
      } catch (err) {
        const errorObj = err instanceof Error ? err : new Error(String(err));
        setError(errorObj);
        setStatus('error');
        throw errorObj;
      }
    },
    [config]
  );

  useEffect(() => {
    if (autoInitialize && status === 'idle') {
      initialize().catch(() => {
        // Error state already captured in state
      });
    }
  }, [autoInitialize, status, initialize]);

  const value: RustoContextValue = useMemo(
    () => ({
      status,
      isReady: status === 'ready',
      error,
      initialize,
      service: defaultOcrService,
    }),
    [status, error, initialize]
  );

  return <RustoContext.Provider value={value}>{children}</RustoContext.Provider>;
}
