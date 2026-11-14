import type { OcrConfig } from './ocrConfig';

export type RustoPresetId = 'ppv6' | 'indra' | 'ppv5' | 'ppv4';

export interface OcrProfilePreset {
  id: RustoPresetId;
  name: string;
  description: string;
  modelId: string;
  config: Partial<OcrConfig>;
}

export const ocrProfilePresets: OcrProfilePreset[] = [
  {
    id: 'ppv6',
    name: 'PP-OCRv6',
    description: 'Default bilingual pipeline (min 960px, slow box scoring, dilation enabled).',
    modelId: 'ppocrv6-tiny',
    config: {
      detectionLimitSideLen: 960,
      detectionLimitType: 'min',
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.6,
      postprocessUnclipRatio: 2.0,
      postprocessUseDilation: true,
      recBatchNum: 1,
      textScore: 0.5,
      lineYThreshold: 0.5,
      wordXThreshold: 0.4,
      calibrationEnabled: false,
      optimizationEnabled: false,
      layoutReading: false,
      enableReadingOrder: false,
    },
  },
  {
    id: 'indra',
    name: 'Indra OCR',
    description: 'Fast Latin & document preset (max 960px, threshold 0.5, batch 1).',
    modelId: 'indraocr',
    config: {
      detectionLimitSideLen: 960,
      detectionLimitType: 'max',
      postprocessThreshold: 0.5,
      postprocessBoxThreshold: 0.0,
      postprocessUnclipRatio: 2.0,
      postprocessUseDilation: true,
      recBatchNum: 1,
      textScore: 0.5,
      lineYThreshold: 0.5,
      wordXThreshold: 0.4,
      calibrationEnabled: false,
      optimizationEnabled: false,
      layoutReading: false,
      enableReadingOrder: false,
    },
  },
  {
    id: 'ppv5',
    name: 'PP-OCRv5',
    description: 'Mobile document preset (min 960px, box threshold 0.5, dilation enabled).',
    modelId: 'ppocrv5-mobile',
    config: {
      detectionLimitSideLen: 960,
      detectionLimitType: 'min',
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.5,
      postprocessUnclipRatio: 2.0,
      postprocessUseDilation: true,
      recBatchNum: 1,
      textScore: 0.5,
      lineYThreshold: 0.5,
      wordXThreshold: 0.4,
      calibrationEnabled: false,
      optimizationEnabled: false,
      layoutReading: false,
      enableReadingOrder: false,
    },
  },
  {
    id: 'ppv4',
    name: 'PP-OCRv4',
    description: 'Legacy high-res preset (max 960px, unclip 1.5, no dilation).',
    modelId: 'ppocrv4-mobile',
    config: {
      detectionLimitSideLen: 960,
      detectionLimitType: 'max',
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.6,
      postprocessUnclipRatio: 1.5,
      postprocessUseDilation: false,
      recBatchNum: 1,
      textScore: 0.5,
      lineYThreshold: 0.5,
      wordXThreshold: 0.4,
      calibrationEnabled: false,
      optimizationEnabled: false,
      layoutReading: false,
      enableReadingOrder: false,
    },
  },
];

export function getPresetById(id: RustoPresetId): OcrProfilePreset {
  return ocrProfilePresets.find((p) => p.id === id) ?? ocrProfilePresets[0]!;
}
