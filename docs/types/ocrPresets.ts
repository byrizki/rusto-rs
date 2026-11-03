import type { OcrConfig } from './ocrConfig';

export interface OcrProfilePreset {
  id: string;
  name: string;
  description: string;
  config: Partial<OcrConfig>;
}

export const ocrProfilePresets: OcrProfilePreset[] = [
  {
    id: 'id-card',
    name: 'ID Card / Form',
    description: 'Background descreening and 960px limit tuned for ID cards, KTP, and passports.',
    config: {
      textScore: 0.5,
      detectionLimitSideLen: 960,
      detectionLimitType: 'max',
      calibrationEnabled: true,
      descreenBackground: true,
      descreenStrength: 1.15,
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.5,
      postprocessUnclipRatio: 1.8,
    },
  },
  {
    id: 'clean-document',
    name: 'Clean Document',
    description: 'Standard document pipeline for high-contrast invoices, receipts, and articles.',
    config: {
      textScore: 0.5,
      detectionLimitSideLen: 960,
      detectionLimitType: 'max',
      calibrationEnabled: false,
      descreenBackground: false,
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.5,
      postprocessUnclipRatio: 1.8,
    },
  },
  {
    id: 'dense-newspaper',
    name: 'Dense Newspaper',
    description:
      'High-resolution detection (1920px) tuned for dense multi-column historical print (ChronAm).',
    config: {
      textScore: 0.45,
      detectionLimitSideLen: 1920,
      detectionLimitType: 'max',
      enableReadingOrder: true,
      layoutModel: 'indra',
      calibrationEnabled: true,
      descreenBackground: true,
      descreenStrength: 1.0,
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.5,
      postprocessUnclipRatio: 1.8,
      postprocessMaxCandidates: 2000,
      optimizationEnabled: true,
      optimizationTargetMaxSide: 1920,
    },
  },
  {
    id: 'dense-faint',
    name: 'Dense / Faint',
    description: 'Lower thresholds and expanded search for thermal slips or small faint text.',
    config: {
      textScore: 0.35,
      detectionLimitSideLen: 1280,
      detectionLimitType: 'max',
      calibrationEnabled: true,
      descreenBackground: true,
      descreenStrength: 1.15,
      postprocessThreshold: 0.2,
      postprocessBoxThreshold: 0.4,
      postprocessUnclipRatio: 2.0,
      postprocessMaxCandidates: 1500,
    },
  },
  {
    id: 'high-precision',
    name: 'High Precision',
    description:
      'Strict confidence cutoffs and tight bounding contours to minimize false positives.',
    config: {
      textScore: 0.65,
      detectionLimitSideLen: 1280,
      detectionLimitType: 'max',
      calibrationEnabled: false,
      descreenBackground: false,
      postprocessThreshold: 0.35,
      postprocessBoxThreshold: 0.6,
      postprocessUnclipRatio: 1.6,
    },
  },
  {
    id: 'fast-mobile',
    name: 'Fast Mobile',
    description:
      'Downscaled detector limit (640px) and fast inference for resource-constrained browsers.',
    config: {
      textScore: 0.5,
      detectionLimitSideLen: 640,
      detectionLimitType: 'max',
      calibrationEnabled: false,
      descreenBackground: false,
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.5,
      postprocessUnclipRatio: 1.8,
      postprocessMaxCandidates: 500,
      optimizationEnabled: true,
      optimizationTargetMaxSide: 640,
    },
  },
];
