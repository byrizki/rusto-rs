import { createContext } from 'react';
import type { RustoContextValue } from '../types/index.js';

export const RustoContext = createContext<RustoContextValue | null>(null);
