import type { InitializeConfig } from 'rusto-web';
import type { OcrService } from '../services/ocrService.js';

export type RustoStatus = 'idle' | 'initializing' | 'ready' | 'error';

export interface RustoContextValue {
  status: RustoStatus;
  isReady: boolean;
  error: Error | null;
  initialize: (config?: InitializeConfig) => Promise<void>;
  service: OcrService;
}

export interface RustoProviderProps {
  children: React.ReactNode;
  config?: InitializeConfig;
  autoInitialize?: boolean;
}
