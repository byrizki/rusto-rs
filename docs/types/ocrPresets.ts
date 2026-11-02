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
      calibrationEnabled: false,
      descreenBackground: false,
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.5,
      postprocessUnclipRatio: 1.8,
    },
  },
  {
    id: 'dense-faint',
    name: 'Dense / Faint',
    description: 'Lower thresholds and expanded search for thermal slips or small faint text.',
    config: {
      textScore: 0.35,
      detectionLimitSideLen: 1280,
      calibrationEnabled: true,
      descreenBackground: true,
      descreenStrength: 1.15,
      postprocessThreshold: 0.2,
      postprocessBoxThreshold: 0.4,
      postprocessUnclipRatio: 2.0,
    },
  },
];
