import { ref, type Ref, type ComputedRef } from 'vue';
import { detectText, isInitialized, type TextResult } from '@rustors/web';
import {
  availableModels,
  defaultModel,
  type LayoutModelId,
  type OcrModelOption,
} from '../types/ocrModels';
import {
  defaultProgressState,
  buildDetectTextOptions,
  type ActiveRunConfig,
  type ProgressState,
  type ProgressStep,
  type OcrConfig,
} from '../types/ocrConfig';
import { loadAndInitModel } from '../utils/ocrModelLoader';
import { isModelSupportedByBackend } from '../utils/ocrReleaseModels';
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
  layoutModelId?: LayoutModelId;
}

function createOcrExecutionState(defaultModelName: string) {
  return {
    progress: ref<ProgressState>({ ...defaultProgressState }),
    activeLoadedModel: ref<string | null>(null),
    activeLoadedLayoutModel: ref<LayoutModelId | null>(null),
    activeLoadedBackend: ref<'rusto-nn' | 'rusto-rten' | 'rusto-webgpu' | null>(null),
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

async function runInitSequence(
  baseURL: string,
  modelMeta: OcrModelOption,
  state: OcrState,
  setProgress: ProgressCallback,
  opts?: EngineInitOptions,
  backend: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu' = 'rusto-nn'
): Promise<void> {
  const isPipeline = opts?.keepProgress ?? false;
  const range: [number, number] = opts?.progressRange ?? (isPipeline ? [0, 60] : [0, 100]);
  state.status.value = 'initializing';
  state.statusMessage.value = `Loading ${modelMeta.name} (${backend})...`;
  setProgress(
    'download',
    `Downloading ${modelMeta.name}...`,
    range[0] + 5,
    'Connecting...',
    opts?.step
  );
  await loadAndInitModel(
    baseURL,
    modelMeta,
    range,
    (prog) => setProgress('download', prog.stageText, prog.percent, prog.detailText, opts?.step),
    opts?.layoutModelId,
    backend
  );
  setProgress(
    'init',
    'Configuring inference engine...',
    isPipeline ? 70 : 100,
    'Ready',
    isPipeline && opts?.step ? { current: 2, total: opts.step.total } : undefined
  );
}

async function performEngineInit(
  baseURL: string,
  modelId: string,
  state: OcrState,
  setProgress: ProgressCallback,
  opts?: EngineInitOptions,
  backend: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu' = 'rusto-nn'
): Promise<boolean> {
  const layoutId = opts?.layoutModelId ?? null;
  if (
    isInitialized() &&
    state.activeLoadedModel.value === modelId &&
    state.activeLoadedLayoutModel.value === layoutId &&
    state.activeLoadedBackend.value === backend
  ) {
    state.status.value = 'ready';
    return true;
  }
  const modelMeta = availableModels.find((m) => m.id === modelId) ?? defaultModel;
  try {
    await runInitSequence(baseURL, modelMeta, state, setProgress, opts, backend);
    state.activeLoadedModel.value = modelId;
    state.activeLoadedLayoutModel.value = layoutId;
    state.activeLoadedBackend.value = backend;
    state.status.value = 'ready';
    state.statusMessage.value = 'Engine Ready';
    return true;
  } catch (err: unknown) {
    state.status.value = 'error';
    const msg = `Failed to load ${modelMeta.name}: ${err instanceof Error ? err.message : String(err)}`;
    state.statusMessage.value = msg;
    state.ocrError.value = msg;
    return false;
  } finally {
    if (!opts?.keepProgress) state.progress.value.active = false;
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
  setProgress('inference', 'Running OCR Inference...', step ? 75 : 25, 'Detecting...', step);
  await new Promise<void>((r) => {
    setTimeout(r, 20);
  });
  const startTime = performance.now();
  setProgress('inference', 'Running OCR Inference...', step ? 88 : 65, 'Decoding...', step);
  const opts = buildDetectTextOptions(ocrConfig, outputMode);
  const builder = detectText(imageSrc).withOptions(opts);
  const output = await (outputMode === 'words' ? builder.asWords() : builder.asLines()).run();
  const durationMs = Math.round(performance.now() - startTime);
  setProgress('inference', 'Finalizing OCR Output...', 100, 'Formatting...', step);
  return { output, durationMs, opts };
}

interface PipelineContext extends WebOcrExecutionOptions {
  state: OcrState;
  initEngine: (modelId?: string, opts?: EngineInitOptions) => Promise<boolean>;
  setProgress: ProgressCallback;
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
  const isReadingOrderActive =
    ctx.ocrConfig.value.layoutReading ||
    ctx.ocrConfig.value.enableReadingOrder ||
    (ctx.ocrConfig.value.layoutModel !== undefined && ctx.ocrConfig.value.layoutModel !== 'none');
  const processed = processOcrOutput(
    res.output,
    ctx.ocrConfig.value.lineYThreshold,
    ctx.ocrConfig.value.wordXThreshold,
    isReadingOrderActive
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
  const currentLayout = ocrConfig.value.layoutModel;
  const isLayoutSynced =
    !currentLayout ||
    currentLayout === 'none' ||
    currentLayout === 'heuristic' ||
    state.activeLoadedLayoutModel.value === currentLayout;
  const needsInit =
    !isInitialized() ||
    state.activeLoadedModel.value !== selectedModel.value ||
    !isLayoutSynced ||
    state.activeLoadedBackend.value !== ocrConfig.value.inferenceBackend;
  try {
    if (needsInit) {
      const ok = await initEngine(selectedModel.value, {
        keepProgress: true,
        step: { current: 1, total: 3 },
        progressRange: [0, 60],
        layoutModelId: currentLayout,
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
  const { baseURL, ocrConfig, selectedModel, currentModel } = options;
  const state = createOcrExecutionState(currentModel.value.name);

  const setProgress: ProgressCallback = (phase, stageText, percent, detailText, step) => {
    state.progress.value = { active: true, phase, stageText, percent, detailText, step };
  };

  const resetResults = () => resetOcrResultsState(state);

  const initEngine = (modelId = selectedModel.value, opts?: EngineInitOptions) =>
    performEngineInit(baseURL, modelId, state, setProgress, opts, ocrConfig.value.inferenceBackend);

  const switchModel = (modelId: string) => {
    selectedModel.value = modelId;
    const modelMeta = availableModels.find((m) => m.id === modelId) ?? defaultModel;
    ocrConfig.value.recBatchNum = modelMeta.recommendedBatchSize;
    let switchedBackendNote = '';
    if (!isModelSupportedByBackend(modelId, ocrConfig.value.inferenceBackend)) {
      ocrConfig.value.inferenceBackend = 'rusto-nn';
      ocrConfig.value.useWebGpu = false;
      switchedBackendNote = ' · switched engine to rusto-nn';
    }
    resetResults();
    const isLoaded =
      state.activeLoadedModel.value === modelId &&
      state.activeLoadedLayoutModel.value === (ocrConfig.value.layoutModel ?? null) &&
      state.activeLoadedBackend.value === ocrConfig.value.inferenceBackend;
    state.status.value = isLoaded ? 'ready' : 'idle';
    state.statusMessage.value = isLoaded
      ? 'Engine Ready'
      : `${currentModel.value.name} selected${switchedBackendNote} (click Run OCR to load)`;
  };

  const runOcr = () =>
    performOcrPipeline({
      ...options,
      state,
      initEngine,
      setProgress,
    });

  return { ...state, initEngine, switchModel, resetResults, runOcr };
}
