import type { OcrSettingsTab } from '../composables/useWebOcrSettings';
import type { OcrConfig } from '../types/ocrConfig';

export interface OcrSettingSearchEntry {
  id: string;
  label: string;
  identifier: string;
  description: string;
  tab: OcrSettingsTab;
  subTab: string;
  breadcrumb: string;
  keywords: string[];
  getValue: (config: OcrConfig) => string;
}

const STAGES = {
  detPre: { tab: 'detection', subTab: 'pre', breadcrumb: 'Detection › Pre-Detection' },
  detCore: { tab: 'detection', subTab: 'core', breadcrumb: 'Detection › Text Detection' },
  detPost: { tab: 'detection', subTab: 'post', breadcrumb: 'Detection › Post-Detection' },
  recPre: { tab: 'recognition', subTab: 'pre', breadcrumb: 'Recognition › Pre-Recognition' },
  recCore: { tab: 'recognition', subTab: 'core', breadcrumb: 'Recognition › Text Recognition' },
  recPost: { tab: 'recognition', subTab: 'post', breadcrumb: 'Recognition › Post-Recognition' },
  rtEngines: { tab: 'runtime', subTab: 'engines', breadcrumb: 'Runtime › Engine Backend' },
  rtModelSelect: { tab: 'runtime', subTab: 'engines', breadcrumb: 'Runtime › Model Selection' },
  rtModel: { tab: 'runtime', subTab: 'model', breadcrumb: 'Runtime › Model Info' },
  rtWeights: { tab: 'runtime', subTab: 'weights', breadcrumb: 'Runtime › Weights & Assets' },
} as const;

type StageMeta = (typeof STAGES)[keyof typeof STAGES];

function entry(
  stage: StageMeta,
  id: string,
  label: string,
  identifier: string,
  description: string,
  keywords: string[],
  getValue: (config: OcrConfig) => string
): OcrSettingSearchEntry {
  return { id, label, identifier, description, ...stage, keywords, getValue };
}

const toggleText = (enabled: boolean) => (enabled ? 'Enabled' : 'Disabled');

export const ocrSettingsSearchIndex: OcrSettingSearchEntry[] = [
  entry(
    STAGES.detPre,
    'autoUnwarp',
    '3D Document Auto-Unwarp',
    '.withAutoUnwarp()',
    'Flattens curved pages, crumpled paper, and folded receipts via 3D UVDoc mesh.',
    ['unwarp', 'uvdoc', 'curve', 'flatten', 'rectify'],
    (c) => toggleText(c.autoUnwarp)
  ),
  entry(
    STAGES.detPre,
    'orientationDetection',
    'Page Orientation Correction',
    '.withOrientationDetection()',
    'Detects and rotates 90°, 180°, and 270° rotated documents.',
    ['orientation', 'rotate', '90', '180', '270'],
    (c) => toggleText(c.orientationDetection)
  ),
  entry(
    STAGES.detPre,
    'calibrationEnabled',
    'Adaptive Lighting Calibration',
    'calibrationEnabled',
    'Normalizes uneven illumination, flash glare, and shadow gradients.',
    ['lighting', 'calibration', 'glare', 'shadow'],
    (c) => toggleText(c.calibrationEnabled)
  ),
  entry(
    STAGES.detPre,
    'descreenBackground',
    'Descreen Patterned Backgrounds',
    'descreenBackground',
    'Suppresses guilloche security patterns and halftone dots on IDs & passports.',
    ['descreen', 'halftone', 'passport', 'id'],
    (c) => (c.calibrationEnabled ? toggleText(c.descreenBackground) : 'Inactive (calibration off)')
  ),
  entry(
    STAGES.detPre,
    'descreenStrength',
    'Descreen Filtering Strength',
    'descreenStrength',
    'Controls intensity of Gaussian & median descreening passes.',
    ['descreen', 'strength', 'gaussian', 'median'],
    (c) =>
      c.calibrationEnabled && c.descreenBackground ? c.descreenStrength.toFixed(2) : 'Inactive'
  ),
  entry(
    STAGES.detPre,
    'optimizationEnabled',
    'SIMD Fast Downscale & Crop',
    'optimizationEnabled',
    'Pre-scales high-resolution camera photos to reduce memory and latency.',
    ['downscale', 'simd', 'optimization', 'targetMaxSide', 'cropPadding'],
    (c) => toggleText(c.optimizationEnabled)
  ),
  entry(
    STAGES.detPre,
    'optimizationTargetMaxSide',
    'Target Max Side',
    'targetMaxSide',
    'Maximum longest side dimension (px) when fast downscale optimization is enabled.',
    ['max side', 'downscale', 'resolution', 'px'],
    (c) => (c.optimizationEnabled ? `${c.optimizationTargetMaxSide}px` : 'Inactive')
  ),
  entry(
    STAGES.detPre,
    'cropPadding',
    'Crop Padding X / Y',
    'cropPaddingX / cropPaddingY',
    'Horizontal (px) and vertical (ratio) crop padding when optimization is enabled.',
    ['padding', 'crop', 'margin'],
    (c) => (c.optimizationEnabled ? `X=${c.cropPaddingX}px, Y=${c.cropPaddingY}` : 'Inactive')
  ),
  entry(
    STAGES.detCore,
    'detectionLimitSideLen',
    'Detector Input Limit',
    'limitSideLen / limitType',
    'Constrains input dimensions fed into DBNet detection model.',
    ['limitSideLen', 'limitType', 'resolution', 'dbnet', 'min', 'max'],
    (c) => `${c.detectionLimitSideLen}px (${c.detectionLimitType})`
  ),
  entry(
    STAGES.detPost,
    'postprocessThreshold',
    'Pixel Binarization',
    'threshold',
    'Probability cutoff for classifying a pixel as text vs background during DBNet binarization.',
    ['threshold', 'binarization', 'pixel', 'dbnet'],
    (c) => c.postprocessThreshold.toFixed(2)
  ),
  entry(
    STAGES.detPost,
    'postprocessBoxThreshold',
    'Box Confidence',
    'boxThreshold',
    'Minimum average score across a detected polygon required to accept a text box.',
    ['boxThreshold', 'confidence', 'polygon', 'score'],
    (c) => c.postprocessBoxThreshold.toFixed(2)
  ),
  entry(
    STAGES.detPost,
    'postprocessUnclipRatio',
    'Polygon Expansion',
    'unclipRatio',
    'Vatti clipping expansion factor used to restore shrunk DBNet regions.',
    ['unclip', 'expansion', 'vatti', 'polygon'],
    (c) => c.postprocessUnclipRatio.toFixed(1)
  ),
  entry(
    STAGES.detPost,
    'postprocessMaxCandidates',
    'Max Candidates',
    'maxCandidates',
    'Maximum number of candidate polygons evaluated per page.',
    ['candidates', 'limit', 'boxes'],
    (c) => String(c.postprocessMaxCandidates)
  ),
  entry(
    STAGES.detPost,
    'postprocessUseDilation',
    'Morphological Dilation',
    'useDilation',
    'Applies 2x2 morphological dilation to connect broken character fragments.',
    ['dilation', 'morphology', '2x2'],
    (c) => toggleText(c.postprocessUseDilation)
  ),
  entry(
    STAGES.detPost,
    'enableResize',
    'Custom Bounding Box Geometry Filters',
    'enableResize',
    'Filter out candidate boxes outside min/max pixel dimensions or aspect ratio.',
    ['geometry', 'resize', 'maxSideLen', 'minSideLen', 'minHeight', 'widthHeightRatio'],
    (c) => (c.enableResize ? `Enabled (${c.minSideLen}–${c.maxSideLen}px)` : 'Disabled')
  ),
  entry(
    STAGES.recPre,
    'layoutModel',
    'Document Layout & Reading Order',
    '.withLayoutReading()',
    'Selects geometric heuristic column flow, neural layout models, or raster scan.',
    ['layout', 'reading order', 'indra', 'cdla', 'publaynet', 'heuristic'],
    (c) => c.layoutModel
  ),
  entry(
    STAGES.recPre,
    'lineThresholds',
    'Line Y Tolerance & Word X Gap',
    'lineYThreshold / wordXThreshold',
    'Heuristic vertical line grouping and horizontal word proximity thresholds.',
    ['lineYThreshold', 'wordXThreshold', 'heuristic', 'spacing', 'gap'],
    (c) =>
      c.layoutModel === 'heuristic'
        ? `Y=${c.lineYThreshold.toFixed(2)}, X=${c.wordXThreshold.toFixed(2)}`
        : 'Inactive (non-heuristic)'
  ),
  entry(
    STAGES.recPre,
    'lineClassification',
    'Upside-Down Line Classifier',
    '.withLineClassification()',
    'Identifies and flips 180° inverted text line crops before recognition.',
    ['cls', '180', 'upside down', 'line classification'],
    (c) => toggleText(c.lineClassification)
  ),
  entry(
    STAGES.recCore,
    'recBatchNum',
    'Recognition Batch Size',
    'batchSize',
    'Number of cropped text lines decoded per inference batch.',
    ['batch', 'batchSize', 'recBatchNum', 'concurrency'],
    (c) => String(c.recBatchNum)
  ),
  entry(
    STAGES.recPost,
    'textScore',
    'Minimum Confidence Cutoff',
    'scoreThreshold',
    'Filters out low-confidence recognition sequences below this threshold.',
    ['score', 'confidence', 'cutoff', 'filter'],
    (c) => `${Math.round(c.textScore * 100)}% (${c.textScore.toFixed(2)})`
  ),
  entry(
    STAGES.rtEngines,
    'inferenceBackend',
    'Inference Engine Backend',
    'rusto-nn / rusto-rten / rusto-webgpu',
    'Selects WebAssembly runtime: pure-Rust MNN, RTen, or WebGPU shaders.',
    ['backend', 'engine', 'wasm', 'webgpu', 'mnn', 'rten'],
    (c) => c.inferenceBackend
  ),
  entry(
    STAGES.rtModelSelect,
    'modelSelection',
    'Model Selection',
    'model',
    'Choose PP-OCRv6 (FP32/FP16/INT8), Indra, PP-OCRv5/v4, or multilingual model.',
    ['model', 'fp16', 'int8', 'fp32', 'indra', 'ppocrv6', 'multilingual'],
    () => 'Select'
  ),
  entry(
    STAGES.rtModel,
    'modelInfo',
    'Model Info & Specs',
    'model',
    'Active model ID, architecture preset, weight size, and recommended batch.',
    ['model', 'architecture', 'fp16', 'int8', 'fp32', 'size'],
    () => 'Inspect'
  ),
  entry(
    STAGES.rtWeights,
    'weightsAssets',
    'Weights & Asset Endpoints',
    'huggingface',
    'Resolved detection, recognition, and dictionary download URLs.',
    ['weights', 'url', 'huggingface', 'cdn', 'dict'],
    () => 'URLs'
  ),
];

export function filterSettingsIndex(query: string): OcrSettingSearchEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return ocrSettingsSearchIndex.filter(
    (item) =>
      item.label.toLowerCase().includes(q) ||
      item.identifier.toLowerCase().includes(q) ||
      item.description.toLowerCase().includes(q) ||
      item.breadcrumb.toLowerCase().includes(q) ||
      item.keywords.some((k) => k.toLowerCase().includes(q))
  );
}
