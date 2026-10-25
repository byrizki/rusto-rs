import { useContext } from 'react';
import { RustoContext } from '../context/RustoContext.js';
import type { RustoContextValue } from '../types/index.js';

/**
 * Access the active RustO OCR context, status, and service instance.
 *
 * @throws Error if called outside of `<RustoProvider>`.
 * @returns The active RustoContextValue.
 */
export function useRusto(): RustoContextValue {
  const context = useContext(RustoContext);
  if (!context) {
    throw new Error('useRusto must be used within a RustoProvider.');
  }
  return context;
}
