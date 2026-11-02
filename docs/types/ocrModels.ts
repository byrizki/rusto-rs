import type { ModelPreset } from '@rustors/web';

export type ModelCategory = 'flagship' | 'ppocrv6' | 'ppocrv5' | 'ppocrv4' | 'multilingual';

export interface ModelCategoryMeta {
  id: ModelCategory;
  name: string;
  description: string;
}

export const modelCategories: ModelCategoryMeta[] = [
  {
    id: 'flagship',
    name: 'Flagship Architecture',
    description: 'Indra OCR DataLab Linenet with 1-channel low-latency recognition',
  },
  {
    id: 'ppocrv6',
    name: 'PP-OCRv6 Series',
    description: 'Latest ultra-lightweight and high-precision variants',
  },
  {
    id: 'ppocrv5',
    name: 'PP-OCRv5 Series',
    description: 'Multilingual mobile and server production models',
  },
  {
    id: 'ppocrv4',
    name: 'PP-OCRv4 Series',
    description: 'Standard high-efficiency baseline models',
  },
  {
    id: 'multilingual',
    name: 'Multilingual Extensions',
    description: 'Script-specific recognition models (Japanese, Korean, Arabic, etc.)',
  },
];

export interface OcrModelOption {
  id: string;
  name: string;
  preset: ModelPreset;
  category: ModelCategory;
  categoryLabel: string;
  size: string;
  bytes?: number;
  trait: string;
  description: string;
  recommendedBatchSize: number;
  tags?: string[];
}

export { availableModels, defaultModel } from './models/ocrModelList';
