import type { OcrConfig } from './ocrConfig';

export type PresetCategory =
  | 'Identity & Finance'
  | 'Documents & Publishing'
  | 'Performance & Camera';

export interface OcrProfilePreset {
  id: string;
  name: string;
  category: PresetCategory;
  badge?: string;
  description: string;
  keywords?: string[];
  config: Partial<OcrConfig>;
}

export const ocrProfilePresets: OcrProfilePreset[] = [
  {
    id: 'id-card',
    name: 'ID Card & Passport',
    category: 'Identity & Finance',
    badge: 'Default',
    description:
      'Calibrated background descreening (1.15) and 960px limit tuned for ID cards, KTP, and passports.',
    keywords: ['ktp', 'id', 'card', 'passport', 'sim', 'license', 'nik', 'descreen'],
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
    id: 'receipt-thermal',
    name: 'Receipt & Thermal Slip',
    category: 'Identity & Finance',
    badge: 'Faint Text',
    description:
      'Lower binarization threshold (0.22) and expanded unclip ratio (2.0) for thermal POS paper.',
    keywords: ['receipt', 'thermal', 'slip', 'pos', 'bill', 'faint', 'invoice', 'store'],
    config: {
      textScore: 0.35,
      detectionLimitSideLen: 1280,
      detectionLimitType: 'max',
      calibrationEnabled: true,
      descreenBackground: true,
      descreenStrength: 1.15,
      postprocessThreshold: 0.22,
      postprocessBoxThreshold: 0.42,
      postprocessUnclipRatio: 2.0,
      postprocessMaxCandidates: 1500,
    },
  },
  {
    id: 'structured-form',
    name: 'Structured Form & Table',
    category: 'Identity & Finance',
    badge: 'Tabular',
    description: 'Calibrated word gap (0.25) and line tolerance (0.35) to maintain tabular fields.',
    keywords: ['form', 'table', 'sheet', 'box', 'fields', 'structured', 'grid', 'tax'],
    config: {
      textScore: 0.5,
      lineYThreshold: 0.35,
      wordXThreshold: 0.25,
      enableReadingOrder: true,
      layoutModel: 'heuristic',
      detectionLimitSideLen: 1280,
      detectionLimitType: 'max',
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.5,
      postprocessUnclipRatio: 1.7,
    },
  },
  {
    id: 'clean-document',
    name: 'Clean Document / Scan',
    category: 'Documents & Publishing',
    badge: 'Fast Scan',
    description:
      'Zero-overhead pipeline for digital PDFs, clean invoices, printed letters, and articles.',
    keywords: ['document', 'pdf', 'scan', 'article', 'invoice', 'clean', 'standard'],
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
    name: 'Dense Multi-Column Print',
    category: 'Documents & Publishing',
    badge: '1920px',
    description:
      'High-resolution detection (1920px) with neural layout order for historical newspapers.',
    keywords: ['newspaper', 'chronam', 'columns', 'multi-column', 'archive', 'dense', 'layout'],
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
    id: 'book-curved',
    name: 'Bound Book & Curved Page',
    category: 'Documents & Publishing',
    badge: 'Unwarp 3D',
    description:
      'Enables 3D auto-unwarp rectification, 180° line classifier, and column flow for thick books.',
    keywords: ['book', 'curved', 'spine', 'page', 'magazine', 'unwarp', 'rotation', 'angle'],
    config: {
      autoUnwarp: true,
      orientationDetection: true,
      lineClassification: true,
      enableReadingOrder: true,
      layoutModel: 'heuristic',
      detectionLimitSideLen: 1280,
      detectionLimitType: 'max',
      postprocessThreshold: 0.3,
      postprocessBoxThreshold: 0.5,
      postprocessUnclipRatio: 1.8,
    },
  },
  {
    id: 'fast-mobile',
    name: 'Fast Mobile / Edge',
    category: 'Performance & Camera',
    badge: 'Low RAM',
    description:
      'Downscaled 640px detector input for instant execution on mobile and low-memory devices.',
    keywords: ['mobile', 'fast', 'edge', 'speed', 'phone', 'lightweight', '640px'],
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
  {
    id: 'high-precision',
    name: 'Maximum Precision',
    category: 'Performance & Camera',
    badge: 'Strict',
    description:
      'Strict confidence cutoffs (0.65) and tight contours (1.6) to prevent false positives.',
    keywords: ['precision', 'strict', 'accurate', 'clean', 'cutoffs', 'anti-hallucination'],
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
    id: 'low-light-camera',
    name: 'Low-Light & Glare Photo',
    category: 'Performance & Camera',
    badge: 'Camera',
    description:
      'Adaptive lighting calibration and descreening (1.25) to normalize flash glare and shadows.',
    keywords: ['camera', 'photo', 'shadow', 'glare', 'flash', 'dark', 'lighting', 'uneven'],
    config: {
      textScore: 0.4,
      detectionLimitSideLen: 1280,
      detectionLimitType: 'max',
      calibrationEnabled: true,
      descreenBackground: true,
      descreenStrength: 1.25,
      postprocessThreshold: 0.25,
      postprocessBoxThreshold: 0.45,
      postprocessUnclipRatio: 1.9,
    },
  },
];
