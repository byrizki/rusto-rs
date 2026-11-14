import type { ModelPreset } from '@rustors/web';
import type { ModelCategory, OcrModelOption } from '../ocrModels';
import { multilingualModels } from './ocrMultilingualModels';

function model(
  id: string,
  name: string,
  preset: ModelPreset,
  category: ModelCategory,
  categoryLabel: string,
  size: string,
  bytes: number,
  trait: string,
  description: string,
  recommendedBatchSize: number,
  tags: string[] = []
): OcrModelOption {
  return {
    id,
    name,
    preset,
    category,
    categoryLabel,
    size,
    bytes,
    trait,
    description,
    recommendedBatchSize,
    tags,
  };
}

const coreModels: OcrModelOption[] = [
  // --- Indra OCR (DataLab) ---
  model(
    'indraocr',
    'Indra OCR (Base)',
    'indra',
    'indra',
    'Indra OCR (DataLab)',
    '~5.5 MB',
    5680573,
    'DataLab Linenet (rec_batch=1)',
    '1-channel low-latency pipeline tuned for complex documents, forms, receipts & ID cards.',
    1,
    ['DataLab', 'Low Latency', 'ID & Docs']
  ),
  model(
    'indraocr-cjk',
    'Indra OCR (CJK Script)',
    'indra',
    'indra',
    'Indra OCR (DataLab)',
    '~10.6 MB',
    10811581,
    'Indra CJK Extension',
    'Indra Linenet architecture extended with Chinese, Japanese, and Korean character sets.',
    1,
    ['CJK', 'DataLab']
  ),
  model(
    'indraocr-world',
    'Indra OCR (World Script)',
    'indra',
    'indra',
    'Indra OCR (DataLab)',
    '~6.1 MB',
    6263549,
    'Indra World Extension',
    'Indra Linenet extended with multi-script international character set.',
    1,
    ['Multilingual', 'International']
  ),

  // --- PP-OCRv6 Series ---
  model(
    'ppocrv6-tiny',
    'PP-OCRv6 Tiny (FP32)',
    'ppv6',
    'ppocrv6',
    'PP-OCRv6 Series (Default)',
    '~6.4 MB',
    6386492,
    'Default · Lightweight FP32',
    'Default general-purpose Apache 2.0 model for fast browser, server, and mobile OCR.',
    1,
    ['Default', 'PPv6', 'Fast']
  ),
  model(
    'ppocrv6-tiny-fp16',
    'PP-OCRv6 Tiny (FP16)',
    'ppv6',
    'ppocrv6',
    'PP-OCRv6 Series',
    '~3.1 MB',
    3208688,
    'Half-precision FP16',
    'Smallest PP-OCRv6 footprint (~3.1 MB) with half-precision FP16 MNN/ONNX weights and near-FP32 accuracy (rusto-nn & WebGPU).',
    1,
    ['PPv6', 'FP16', 'Fast']
  ),
  model(
    'ppocrv6-tiny-int8',
    'PP-OCRv6 Tiny (INT8)',
    'ppv6',
    'ppocrv6',
    'PP-OCRv6 Series',
    '~3.5 MB',
    3574900,
    'Ultra-lightweight INT8',
    'Smallest footprint model with INT8 quantized weights for minimal memory consumption (rusto-nn & WebGPU; unsupported on RTen).',
    1,
    ['PPv6', 'INT8']
  ),
  model(
    'ppocrv6-small',
    'PP-OCRv6 Small (FP32)',
    'ppv6',
    'ppocrv6',
    'PP-OCRv6 Series',
    '~31 MB',
    31209516,
    'Balanced Speed & Accuracy',
    'High detection precision with moderate memory overhead.',
    1,
    ['PPv6', 'Balanced']
  ),
  model(
    'ppocrv6-small-int8',
    'PP-OCRv6 Small (INT8)',
    'ppv6',
    'ppocrv6',
    'PP-OCRv6 Series',
    '~16.7 MB',
    16691876,
    'Balanced INT8',
    'Quantized version of PP-OCRv6 Small with half the memory footprint (rusto-nn & WebGPU; unsupported on RTen).',
    1,
    ['PPv6', 'INT8']
  ),
  model(
    'ppocrv6-medium',
    'PP-OCRv6 Medium (FP32)',
    'ppv6',
    'ppocrv6',
    'PP-OCRv6 Series',
    '~139 MB',
    138809964,
    'Highest Accuracy FP32',
    'Deep recognition backbone for challenging irregular text and noisy scans.',
    1,
    ['PPv6', 'High Accuracy']
  ),
  model(
    'ppocrv6-medium-int8',
    'PP-OCRv6 Medium (INT8)',
    'ppv6',
    'ppocrv6',
    'PP-OCRv6 Series',
    '~79 MB',
    79337492,
    'Highest Accuracy INT8',
    'Quantized Medium model balancing high accuracy with reduced bandwidth (rusto-nn & WebGPU; unsupported on RTen).',
    1,
    ['PPv6', 'INT8']
  ),

  // --- PP-OCRv5 Series ---
  model(
    'ppocrv5-mobile',
    'PP-OCRv5 Mobile',
    'ppv5',
    'ppocrv5',
    'PP-OCRv5 Series',
    '~21 MB',
    21554080,
    'Fast Mobile FP32',
    'Proven mobile architecture with broad multilingual vocabulary.',
    1,
    ['PPv5', 'Mobile']
  ),

  // --- PP-OCRv4 Series ---
  model(
    'ppocrv4-mobile',
    'PP-OCRv4 Mobile',
    'ppv4',
    'ppocrv4',
    'PP-OCRv4 Series',
    '~16 MB',
    15705129,
    'Fast & Compact',
    'Widely used lightweight v4 pipeline for fast document parsing.',
    1,
    ['PPv4', 'Mobile']
  ),
];

export const availableModels: OcrModelOption[] = [...coreModels, ...multilingualModels];

export const defaultModel: OcrModelOption = (availableModels.find((m) => m.id === 'ppocrv6-tiny') ??
  availableModels[0]) as OcrModelOption;
