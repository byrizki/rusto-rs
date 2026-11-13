import type { OcrConfig } from './ocrConfig';
import type { RustoPresetId } from './ocrPresets';

export interface SampleImage {
  id: string;
  name: string;
  description: string;
  url: string;
  presetId: RustoPresetId;
  configOverrides?: Partial<OcrConfig>;
}

interface SampleDefinition extends Omit<SampleImage, 'url'> {
  fileName: string;
}

const SAMPLE_DEFINITIONS: SampleDefinition[] = [
  {
    id: 'invoice',
    name: 'Invoice',
    description: 'Structured layout with numbers & tables',
    fileName: 'example1.png',
    presetId: 'ppv6',
  },
  {
    id: 'idcard',
    name: 'Passport',
    description: 'Identity document with personal details & MRZ',
    fileName: 'idcard.jpg',
    presetId: 'indra',
    configOverrides: {
      calibrationEnabled: true,
      descreenBackground: true,
      descreenStrength: 1.15,
      optimizationEnabled: true,
      optimizationTargetMaxSide: 800,
      cropPaddingX: 0,
      cropPaddingY: 0.02,
    },
  },
  {
    id: 'handwritten',
    name: 'Handwritten',
    description: 'Vintage cursive handwriting receipt',
    fileName: 'invoice1.jpg',
    presetId: 'ppv6',
    configOverrides: {
      lineYThreshold: 0.7,
      wordXThreshold: 1.4,
      postprocessUseDilation: true,
    },
  },
  {
    id: 'dense-newspaper',
    name: 'Dense Newspaper',
    description: 'Dense historical multi-column newspaper (ChronAm)',
    fileName: 'dense-newspaper.png',
    presetId: 'ppv6',
    configOverrides: {
      detectionLimitType: 'max',
      detectionLimitSideLen: 1536,
      layoutReading: true,
      enableReadingOrder: true,
      layoutModel: 'heuristic',
    },
  },
];

export function getSampleImages(baseURL: string): SampleImage[] {
  return SAMPLE_DEFINITIONS.map((def) => ({
    id: def.id,
    name: def.name,
    description: def.description,
    url: `${baseURL}samples/${def.fileName}`,
    presetId: def.presetId,
    configOverrides: def.configOverrides,
  }));
}
