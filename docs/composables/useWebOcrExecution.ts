import { ref, type Ref, type ComputedRef } from 'vue';
import { detectText, isInitialized, type TextResult } from '@rustors/web';
import { availableModels, defaultModel, type OcrModelOption } from '../types/ocrModels';
import {
  defaultProgressState,
  buildDetectTextOptions,
  type ActiveRunConfig,
  type ProgressState,
  type ProgressStep,
  type OcrConfig,
} from '../types/ocrConfig';
import { loadAndInitModel } from '../utils/ocrModelLoader';
import { buildActiveRunConfig, processOcrOutput } from '../utils/ocrResultFormatter';

export interface WebOcrExecutionOptions {
  baseURL: string;
  imageSrc: Ref<string>;
  ocrConfig: Ref<OcrConfig>;
  selectedModel: Ref<string>;
  currentModel: ComputedRef<OcrModelOption>;
  outputMode: Ref<'lines' | 'words'>;
}

export type ProgressCallback = (
  phase: 'download' | 'init' | 'inference',
  stageText: string,
  percent: number,
  detailText: string,
  step?: ProgressStep
) => void;

export interface EngineInitOptions {
  keepProgress?: boolean;
  step?: ProgressStep;
  progressRange?: [number, number];
}

function createOcrExecutionState(defaultModelName: string) {
  return {
    progress: ref<ProgressState>({ ...defaultProgressState }),
    activeLoadedModel: ref<string | null>(null),
    status: ref<'idle' | 'initializing' | 'ready' | 'error'>('idle'),
    statusMessage: ref<string>(`${defaultModelName} selected (click Run OCR to load)`),
    isScanning: ref<boolean>(false),
    durationMs: ref<number | null>(null),
    activeRunConfig: ref<ActiveRunConfig | null>(null),
    results: ref<TextResult[]>([]),
    spatialText: ref<string>(''),
    hoveredIndex: ref<number | null>(null),
    ocrError: ref<string | null>(null),
  };
}

type OcrState = ReturnType<typeof createOcrExecutionState>;

function resetOcrResultsState(state: OcrState) {
  state.results.value = [];
  state.spatialText.value = '';
  state.durationMs.value = null;
  state.hoveredIndex.value = null;
  state.activeRunConfig.value = null;
  state.ocrError.value = null;
}

async function loadEngineWeights(
  baseURL: string,
  modelMeta: OcrModelOption,
  range: [number, number],
  setProgress: ProgressCallback,
  step?: ProgressStep
) {
  await loadAndInitModel(baseURL, modelMeta, range, (prog) => {
    setProgress('download', prog.stageText, prog.percent, prog.detailText, step);
  });
}

async function performEngineInit(
  baseURL: string,
  modelId: string,
  state: OcrState,
  setProgress: ProgressCallback,
  opts?: EngineInitOptions
): Promise<boolean> {
  if (isInitialized() && state.activeLoadedModel.value === modelId) {
    state.status.value = 'ready';
    return true;
  }
  const modelMeta = availableModels.find((m) => m.id === modelId) ?? defaultModel;
  const isPipeline = opts?.keepProgress ?? false;
  const step = opts?.step;
  const range: [number, number] = opts?.progressRange ?? (isPipeline ? [0, 60] : [0, 100]);
  try {
    state.status.value = 'initializing';
    state.statusMessage.value = `Loading ${modelMeta.name}...`;
    setProgress(
      'download',
      `Downloading ${modelMeta.name}...`,
      range[0] + 5,
      'Connecting...',
      step
    );
    await loadEngineWeights(baseURL, modelMeta, range, setProgress, step);

    setProgress(
      'init',
      'Configuring inference engine...',
      isPipeline ? 70 : 100,
      'Ready',
      isPipeline && step ? { current: 2, total: step.total } : undefined
    );
    state.activeLoadedModel.value = modelId;
    state.status.value = 'ready';
    state.statusMessage.value = 'Engine Ready';
    return true;
  } catch (err: unknown) {
    state.status.value = 'error';
    const msg = `Failed to load ${modelMeta.name}: ${err instanceof Error ? err.message : String(err)}`;
    state.statusMessage.value = msg;
    state.ocrError.value = msg;
    console.error(`OCR Init Error [${modelMeta.name}]:`, err);
    return false;
  } finally {
    if (!isPipeline) state.progress.value.active = false;
  }
}

async function executeModelInference(
  imageSrc: string,
  ocrConfig: OcrConfig,
  outputMode: 'lines' | 'words',
  setProgress: ProgressCallback,
  step?: ProgressStep
): Promise<{
  output: TextResult[] | string;
  durationMs: number;
  opts: ReturnType<typeof buildDetectTextOptions>;
}> {
  setProgress(
    'inference',
    'Running OCR Inference...',
    step ? 75 : 25,
    'Detecting text regions...',
    step
  );
  await new Promise<void>((resolve) => {
    setTimeout(resolve, 20);
  });
  const startTime = performance.now();
  setProgress(
    'inference',
    'Running OCR Inference...',
    step ? 88 : 65,
    'Decoding characters...',
    step
  );
  const opts = buildDetectTextOptions(ocrConfig, outputMode);
  const output = await detectText(imageSrc, opts);
  const durationMs = Math.round(performance.now() - startTime);
  setProgress('inference', 'Finalizing OCR Output...', 100, 'Formatting results...', step);
  return { output, durationMs, opts };
}

interface PipelineContext {
  state: OcrState;
  selectedModel: Ref<string>;
  initEngine: (modelId?: string, opts?: EngineInitOptions) => Promise<boolean>;
  imageSrc: Ref<string>;
  ocrConfig: Ref<OcrConfig>;
  outputMode: Ref<'lines' | 'words'>;
  setProgress: ProgressCallback;
  currentModel: ComputedRef<OcrModelOption>;
}

function applyProcessedPipelineOutput(
  state: OcrState,
  res: {
    output: TextResult[] | string;
    durationMs: number;
    opts: ReturnType<typeof buildDetectTextOptions>;
  },
  ctx: PipelineContext
) {
  state.durationMs.value = res.durationMs;
  const processed = processOcrOutput(
    res.output,
    ctx.ocrConfig.value.lineYThreshold,
    ctx.ocrConfig.value.wordXThreshold,
    ctx.ocrConfig.value.enableReadingOrder
  );
  state.results.value = processed.results;
  state.spatialText.value = processed.spatialText;
  state.activeRunConfig.value = buildActiveRunConfig(
    ctx.currentModel.value,
    ctx.outputMode.value,
    res.durationMs,
    processed.results.length,
    res.opts,
    ctx.ocrConfig.value
  );
}

async function performOcrPipeline(ctx: PipelineContext) {
  const { state, selectedModel, initEngine, imageSrc, ocrConfig, outputMode, setProgress } = ctx;
  if (state.isScanning.value || state.status.value === 'initializing') return;
  resetOcrResultsState(state);
  state.isScanning.value = true;
  const needsInit = !isInitialized() || state.activeLoadedModel.value !== selectedModel.value;
  try {
    if (needsInit) {
      const ok = await initEngine(selectedModel.value, {
        keepProgress: true,
        step: { current: 1, total: 3 },
        progressRange: [0, 60],
      });
      if (!ok) return;
    }
    const res = await executeModelInference(
      imageSrc.value,
      ocrConfig.value,
      outputMode.value,
      setProgress,
      needsInit ? { current: 3, total: 3 } : undefined
    );
    applyProcessedPipelineOutput(state, res, ctx);
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    state.ocrError.value = errorMsg;
    state.status.value = 'error';
    state.statusMessage.value = `OCR Error: ${errorMsg}`;
    console.error('OCR Detection error:', err);
  } finally {
    state.isScanning.value = false;
    state.progress.value.active = false;
  }
}

export function useWebOcrExecution(options: WebOcrExecutionOptions) {
  const { baseURL, imageSrc, ocrConfig, selectedModel, currentModel, outputMode } = options;
  const state = createOcrExecutionState(currentModel.value.name);

  const setProgress: ProgressCallback = (phase, stageText, percent, detailText, step) => {
    state.progress.value = { active: true, phase, stageText, percent, detailText, step };
  };

  const resetResults = () => resetOcrResultsState(state);

  const initEngine = (modelId = selectedModel.value, opts?: EngineInitOptions) =>
    performEngineInit(baseURL, modelId, state, setProgress, opts);

  const switchModel = (modelId: string) => {
    selectedModel.value = modelId;
    const modelMeta = availableModels.find((m) => m.id === modelId) ?? defaultModel;
    ocrConfig.value.recBatchNum = modelMeta.recommendedBatchSize;
    resetResults();
    const isLoaded = state.activeLoadedModel.value === modelId;
    state.status.value = isLoaded ? 'ready' : 'idle';
    state.statusMessage.value = isLoaded
      ? 'Engine Ready'
      : `${currentModel.value.name} selected (click Run OCR to load)`;
  };

  const runOcr = () =>
    performOcrPipeline({
      state,
      selectedModel,
      initEngine,
      imageSrc,
      ocrConfig,
      outputMode,
      setProgress,
      currentModel,
    });

  return { ...state, initEngine, switchModel, resetResults, runOcr };
}
