// Provider & Context
export { RustoProvider } from './context/RustoProvider.js';
export { RustoContext } from './context/RustoContext.js';

// Hooks
export { useRusto } from './hooks/useRusto.js';
export { useOcr } from './hooks/useOcr.js';

// Services
export { OcrService, defaultOcrService } from './services/ocrService.js';

// UI Components
export { OcrOverlay } from './components/OcrOverlay.js';
export { BoundingBox } from './components/BoundingBox.js';
export { OcrDropzone } from './components/OcrDropzone.js';
export { DropzonePrompt } from './components/DropzonePrompt.js';
export { DropzonePreview } from './components/DropzonePreview.js';

// Geometry & Formatters
export { scaleFrame, scaleQuad, calculateScale } from './utils/geometry.js';
export { resultsToText, resultsToCsv, resultsToJson } from './utils/formatters.js';

// Types
export * from './types/index.js';

// Re-export common types from rusto-web
export type {
  TextResult,
  Frame,
  ImageSource,
  DetectTextOptions,
  InitializeConfig,
  ModelPreset,
} from 'rusto-web';
