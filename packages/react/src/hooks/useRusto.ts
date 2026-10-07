import { useContext } from 'react';
import { RustoContext } from '../context/RustoContext.js';
import type { RustoContextValue } from '../types/index.js';

export function useRusto(): RustoContextValue {
  const context = useContext(RustoContext);
  if (!context) {
    throw new Error('useRusto must be used within a RustoProvider.');
  }
  return context;
}
