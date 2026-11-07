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

export type LayoutModelId = 'heuristic' | 'indra' | 'cdla' | 'publaynet' | 'none';

export interface LayoutModelOption {
  id: LayoutModelId;
  name: string;
  category: 'Indra OCR' | 'PP-Structure' | 'Built-in';
  description: string;
  size: string;
  assetFile?: string;
  downloadUrl?: string;
}

export const availableLayoutModels: LayoutModelOption[] = [
  {
    id: 'heuristic',
    name: 'Geometric Column Flow (Built-in)',
    category: 'Built-in',
    description:
      'Fast client-side geometric column partitioning and reading flow without extra model download.',
    size: '0 KB',
  },
  {
    id: 'indra',
    name: 'Indra Neural Reading Order (Transformer)',
    category: 'Indra OCR',
    description:
      'Sequences multi-column text lines into natural human reading flow using coordinate transformer.',
    size: '2.9 MB',
    assetFile: 'indraocr/indra_order.rten',
    downloadUrl:
      'https://huggingface.co/byrizki/rusto-models/resolve/main/indraocr/indra_order.rten',
  },
  {
    id: 'cdla',
    name: 'PP-OCR CDLA Document Layout',
    category: 'PP-Structure',
    description:
      'General document layout analysis detecting text, titles, figures, captions, tables, and headers.',
    size: '8.4 MB',
    assetFile: 'modules/layout_cdla.rten',
    downloadUrl: 'https://huggingface.co/byrizki/rusto-models/resolve/main/modules/layout_cdla.rten',
  },
  {
    id: 'publaynet',
    name: 'PP-OCR PubLayNet Paper Layout',
    category: 'PP-Structure',
    description:
      'Academic paper and publication layout model detecting text, titles, lists, tables, and figures.',
    size: '8.4 MB',
    assetFile: 'modules/layout_publaynet.rten',
    downloadUrl:
      'https://huggingface.co/byrizki/rusto-models/resolve/main/modules/layout_publaynet.rten',
  },
  {
    id: 'none',
    name: 'None (Linear Raster Scan)',
    category: 'Built-in',
    description: 'Standard horizontal line-by-line raster scan from top to bottom.',
    size: '0 KB',
  },
];

export { availableModels, defaultModel } from './models/ocrModelList';
