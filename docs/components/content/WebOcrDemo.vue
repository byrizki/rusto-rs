<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useRuntimeConfig } from '#app';
import {
  initialize,
  detectText,
  isInitialized,
  formatSpatialText,
  groupCandidatesIntoLines,
  type TextResult,
  type OutputGranularity,
} from 'rusto-web';
import {
  availableModels,
  defaultOcrConfig,
  buildDetectTextOptions,
  type OcrConfig,
} from '../../types/ocrConfig';

const runtimeConfig = useRuntimeConfig();
const rawBaseURL = runtimeConfig.app.baseURL || '/';
const baseURL = rawBaseURL.endsWith('/') ? rawBaseURL : `${rawBaseURL}/`;

// Sample images configuration
const samples = [
  {
    id: 'invoice',
    name: 'Invoice',
    description: 'Structured layout with numbers & tables',
    url: `${baseURL}samples/example1.png`,
  },
  {
    id: 'idcard',
    name: 'Passport',
    description: 'Identity document with personal details & MRZ',
    url: `${baseURL}samples/idcard.jpg`,
  },
  {
    id: 'handwritten',
    name: 'Handwritten',
    description: 'Vintage cursive handwriting receipt',
    url: `${baseURL}samples/invoice1.jpg`,
  },
];

const ocrConfig = ref<OcrConfig>({ ...defaultOcrConfig });
const isConfigOpen = ref<boolean>(false);
const configTab = ref<'detection' | 'layout' | 'resizing' | 'preprocess'>('detection');

function toggleConfigPopover() {
  isConfigOpen.value = !isConfigOpen.value;
}

function resetConfigDefaults() {
  ocrConfig.value = { ...defaultOcrConfig };
}

// Progress State for Download and OCR
interface ProgressState {
  active: boolean;
  phase: 'download' | 'inference';
  stageText: string;
  percent: number;
  detailText: string;
}

const progress = ref<ProgressState>({
  active: false,
  phase: 'download',
  stageText: '',
  percent: 0,
  detailText: '',
});

// Reactive state
const selectedModel = ref<string>('ppocrv6-tiny');
const activeLoadedModel = ref<string | null>(null);
const currentModel = computed(
  () => availableModels.find((m) => m.id === selectedModel.value) || availableModels[0]
);

const selectedSample = ref<string>('invoice');
const imageSrc = ref<string>(samples[0].url);
const status = ref<'idle' | 'initializing' | 'ready' | 'error'>('idle');
const statusMessage = ref<string>('PP-OCRv6 Tiny selected (click Run OCR to load)');
const isScanning = ref<boolean>(false);
const durationMs = ref<number | null>(null);

// Configuration state
const outputMode = ref<OutputGranularity>('lines');
const activeTab = ref<'structured' | 'spatial' | 'csv' | 'json'>('structured');
const copied = ref<boolean>(false);

// Results state
const results = ref<TextResult[]>([]);
const spatialText = ref<string>('');
const hoveredIndex = ref<number | null>(null);

// Image scaling state
const imgRef = ref<HTMLImageElement | null>(null);
const naturalWidth = ref<number>(1);
const naturalHeight = ref<number>(1);
const renderWidth = ref<number>(1);
const renderHeight = ref<number>(1);

const scaleX = computed(() => renderWidth.value / naturalWidth.value || 1);
const scaleY = computed(() => renderHeight.value / naturalHeight.value || 1);

// Download binary model with stream progress
async function downloadBinaryWithProgress(
  url: string,
  label: string,
  stepIndex: number,
  totalSteps: number
): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body?.getReader();
  if (!reader) {
    const ab = await res.arrayBuffer();
    return new Uint8Array(ab);
  }
  let received = 0;
  const chunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const fraction = total > 0 ? received / total : 0.5;
    const overallPct = Math.round(((stepIndex - 1 + fraction) / totalSteps) * 100);
    const mbRec = (received / (1024 * 1024)).toFixed(1);
    const mbTot = total > 0 ? `${(total / (1024 * 1024)).toFixed(1)} MB` : '...';
    progress.value = {
      active: true,
      phase: 'download',
      stageText: `Downloading ${label}`,
      percent: Math.min(overallPct, 99),
      detailText: `${mbRec} / ${mbTot} (${overallPct}%)`,
    };
  }
  const output = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    output.set(c, offset);
    offset += c.length;
  }
  return output;
}

// Download text dictionary asset
async function downloadTextAsset(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
  return await res.text();
}

// Initialize engine with RTen models (no ONNX)
async function initEngine(modelId = selectedModel.value): Promise<boolean> {
  if (isInitialized() && activeLoadedModel.value === modelId) {
    status.value = 'ready';
    return true;
  }
  const modelMeta = availableModels.find((m) => m.id === modelId) || availableModels[0];
  try {
    status.value = 'initializing';
    statusMessage.value = `Loading ${modelMeta.name}...`;

    progress.value = {
      active: true,
      phase: 'download',
      stageText: `Downloading ${modelMeta.name}...`,
      percent: 5,
      detailText: 'Connecting to model repository...',
    };

    const detUrl = `${baseURL}models/${modelMeta.id}/det.rten`;
    const recUrl = `${baseURL}models/${modelMeta.id}/rec.rten`;
    const dictUrl = `${baseURL}models/${modelMeta.id}/dict.txt`;

    const detBuffer = await downloadBinaryWithProgress(
      detUrl,
      `${modelMeta.name} (det.rten)`,
      1,
      2
    );
    const recBuffer = await downloadBinaryWithProgress(
      recUrl,
      `${modelMeta.name} (rec.rten)`,
      2,
      2
    );
    const dictText = await downloadTextAsset(dictUrl);

    progress.value = {
      active: true,
      phase: 'download',
      stageText: 'Initializing RTen WebAssembly Pipeline...',
      percent: 100,
      detailText: 'Configuring inference engine...',
    };

    await initialize({
      preset: modelMeta.preset,
      wasmUrl: `${baseURL}wasm/rusto_rten_wasm_bg.wasm`,
      models: {
        detection: detBuffer,
        recognition: recBuffer,
        dictionary: dictText,
      },
    });

    activeLoadedModel.value = modelId;
    status.value = 'ready';
    statusMessage.value = 'Engine Ready';
    return true;
  } catch (err: unknown) {
    status.value = 'error';
    statusMessage.value = `Failed to load ${modelMeta.name}: ${err instanceof Error ? err.message : String(err)}`;
    console.error(`OCR Init Error [${modelMeta.name}]:`, err);
    return false;
  } finally {
    progress.value.active = false;
  }
}

// Switch active OCR model without downloading immediately
function switchModel(modelId: string) {
  selectedModel.value = modelId;
  resetResults();
  if (activeLoadedModel.value === modelId) {
    status.value = 'ready';
    statusMessage.value = 'Engine Ready';
  } else {
    status.value = 'idle';
    statusMessage.value = `${currentModel.value.name} selected (click Run OCR to load)`;
  }
}

// Switch granularity and align active inspector tab
function setGranularity(mode: OutputGranularity) {
  outputMode.value = mode;
  if (mode === 'spatial') {
    activeTab.value = 'spatial';
  } else {
    activeTab.value = 'structured';
  }
  if (results.value.length > 0 || spatialText.value) {
    runOcr();
  }
}

// Clear results helper
function resetResults() {
  results.value = [];
  spatialText.value = '';
  durationMs.value = null;
  hoveredIndex.value = null;
}

// Run OCR detection manually (downloads model on demand if needed)
async function runOcr() {
  if (isScanning.value || status.value === 'initializing') return;

  // If model is not loaded yet, download and initialize first
  if (!isInitialized() || activeLoadedModel.value !== selectedModel.value) {
    const success = await initEngine(selectedModel.value);
    if (!success) return;
  }

  resetResults();
  isScanning.value = true;
  progress.value = {
    active: true,
    phase: 'inference',
    stageText: 'Running OCR Inference...',
    percent: 25,
    detailText: 'Preprocessing & detecting text boxes...',
  };

  await new Promise<void>((resolve) => {
    setTimeout(resolve, 20);
  });
  const startTime = performance.now();

  try {
    progress.value = {
      active: true,
      phase: 'inference',
      stageText: 'Running OCR Inference...',
      percent: 60,
      detailText: 'Decoding character probabilities...',
    };

    const isSpatial = outputMode.value === 'spatial';
    const opts = buildDetectTextOptions(ocrConfig.value, outputMode.value);

    const output = await detectText(imageSrc.value, opts);
    durationMs.value = Math.round(performance.now() - startTime);

    if (Array.isArray(output)) {
      results.value = output;
      spatialText.value = formatSpatialText(
        groupCandidatesIntoLines(output, ocrConfig.value.lineYThreshold),
        ocrConfig.value.wordXThreshold
      );
    } else {
      spatialText.value = output;
      results.value = [];
    }

    if (isSpatial) {
      activeTab.value = 'spatial';
    }
  } catch (err) {
    console.error('OCR Detection error:', err);
  } finally {
    isScanning.value = false;
    progress.value.active = false;
  }
}

// Handle sample pick without auto-running OCR
function pickSample(sample: (typeof samples)[0]) {
  resetResults();
  selectedSample.value = sample.id;
  imageSrc.value = sample.url;
}

// Handle custom file upload without auto-running OCR
function handleFileUpload(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  resetResults();
  selectedSample.value = 'custom';
  imageSrc.value = `${URL.createObjectURL(file)}#${encodeURIComponent(file.name)}`;
}

// Handle image load event to record natural & displayed dimensions
function onImageLoad(e: Event) {
  const img = e.target as HTMLImageElement;
  naturalWidth.value = img.naturalWidth || 1;
  naturalHeight.value = img.naturalHeight || 1;
  renderWidth.value = img.clientWidth || img.naturalWidth || 1;
  renderHeight.value = img.clientHeight || img.naturalHeight || 1;
}

// Copy results to clipboard
async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    copied.value = true;
    setTimeout(() => {
      copied.value = false;
    }, 2000);
  } catch (err) {
    console.error('Copy failed:', err);
  }
}

// Computed CSV
const csvExport = computed(() => {
  const header = 'text,score,left,top,width,height';
  const rows = results.value.map(
    (r) =>
      `"${r.text.replaceAll('"', '""')}",${r.score.toFixed(4)},${r.frame.left},${r.frame.top},${r.frame.width},${r.frame.height}`
  );
  return [header, ...rows].join('\n');
});

// Computed JSON
const jsonExport = computed(() => JSON.stringify(results.value, null, 2));

onMounted(() => {
  status.value = 'idle';
  statusMessage.value = `${currentModel.value.name} selected (click Run OCR to load)`;
});
</script>

<template>
  <div
    class="rusto-demo-container not-prose my-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 shadow-xl overflow-hidden"
  >
    <!-- Top Action & Status Bar -->
    <div
      class="demo-header flex flex-wrap items-center justify-between gap-3 p-3.5 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950"
    >
      <!-- Left: Engine Status Badge + Model Picker + Trait Aside -->
      <div class="flex flex-wrap items-center gap-3">
        <!-- Status Indicator (Color Only) -->
        <div class="flex items-center gap-1.5" :title="statusMessage">
          <span
            class="inline-block w-2.5 h-2.5 rounded-full shrink-0"
            :class="{
              'bg-emerald-500': status === 'ready',
              'bg-amber-500 animate-pulse': status === 'initializing',
              'bg-red-500': status === 'error',
              'bg-neutral-400': status === 'idle',
            }"
          />
          <button
            v-if="status === 'error'"
            type="button"
            @click="initEngine(selectedModel)"
            class="text-[11px] px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-950 text-red-600 dark:text-red-400 hover:underline cursor-pointer"
          >
            Retry
          </button>
        </div>

        <span class="text-neutral-300 dark:text-neutral-700 hidden sm:inline">|</span>

        <!-- Consolidated Model Picker Select -->
        <div class="inline-flex items-center gap-2">
          <label
            for="ocr-model-select"
            class="text-xs font-semibold text-neutral-500 dark:text-neutral-400"
          >
            Model:
          </label>
          <div class="relative inline-flex items-center">
            <select
              id="ocr-model-select"
              v-model="selectedModel"
              :disabled="status === 'initializing' || isScanning"
              @change="switchModel(selectedModel)"
              class="appearance-none pl-2.5 pr-7 py-1 text-xs font-semibold rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-neutral-800 dark:text-neutral-200 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary disabled:opacity-50 disabled:cursor-not-allowed shadow-xs transition-all cursor-pointer"
            >
              <option v-for="model in availableModels" :key="model.id" :value="model.id">
                {{ model.name }} ({{ model.size }})
              </option>
            </select>
            <div
              class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-1.5 text-neutral-400"
            >
              <svg class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">
                <path
                  fill-rule="evenodd"
                  d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                  clip-rule="evenodd"
                />
              </svg>
            </div>
          </div>

          <!-- Selected Model Trait Put Aside -->
          <span
            class="text-[11px] px-2 py-0.5 rounded-full font-medium bg-neutral-200/80 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 hidden sm:inline"
          >
            {{ currentModel.trait }}
          </span>
        </div>
      </div>

      <!-- Right: Configurator + Duration + Run OCR Button -->
      <div class="flex items-center gap-2">
        <span
          v-if="durationMs !== null"
          class="text-xs text-neutral-500 dark:text-neutral-400 font-mono mr-1"
        >
          {{ durationMs }} ms
        </span>

        <!-- OCR Configurator Popover Button -->
        <div class="relative">
          <button
            type="button"
            @click="toggleConfigPopover"
            class="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-all cursor-pointer shadow-xs"
            :class="{ 'border-primary text-primary': isConfigOpen }"
            title="OCR Engine Settings"
          >
            <svg
              class="w-3.5 h-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <circle cx="12" cy="12" r="3" />
              <path
                d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
              />
            </svg>
            <span class="hidden sm:inline">Settings</span>
          </button>

          <!-- Floating Popover Modal -->
          <div
            v-if="isConfigOpen"
            class="absolute right-0 top-full mt-2 w-84 sm:w-[420px] max-h-[85vh] flex flex-col rounded-xl shadow-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 z-50 text-xs overflow-hidden"
          >
            <!-- Popover Header -->
            <div
              class="flex items-center justify-between p-3.5 border-b border-neutral-200 dark:border-neutral-800 shrink-0"
            >
              <div class="flex items-center gap-2">
                <h4 class="font-semibold text-neutral-800 dark:text-neutral-200">
                  OCR Engine Settings
                </h4>
              </div>
              <div class="flex items-center gap-2">
                <button
                  type="button"
                  @click="resetConfigDefaults"
                  class="text-[11px] text-neutral-500 hover:text-primary transition-colors cursor-pointer"
                >
                  Reset Defaults
                </button>
                <button
                  type="button"
                  @click="isConfigOpen = false"
                  class="text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-200 cursor-pointer text-sm font-bold px-1"
                >
                  ✕
                </button>
              </div>
            </div>

            <!-- Category Tabs -->
            <div class="px-3.5 pt-2.5 pb-1 border-b border-neutral-100 dark:border-neutral-800 shrink-0">
              <div class="flex items-center gap-1 p-1 bg-neutral-100 dark:bg-neutral-800/80 rounded-lg">
                <button
                  type="button"
                  @click="configTab = 'detection'"
                  class="flex-1 py-1 text-[11px] font-medium rounded-md transition-all cursor-pointer text-center"
                  :class="
                    configTab === 'detection'
                      ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-white shadow-xs font-semibold'
                      : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-200'
                  "
                >
                  Detection
                </button>
                <button
                  type="button"
                  @click="configTab = 'layout'"
                  class="flex-1 py-1 text-[11px] font-medium rounded-md transition-all cursor-pointer text-center"
                  :class="
                    configTab === 'layout'
                      ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-white shadow-xs font-semibold'
                      : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-200'
                  "
                >
                  Layout
                </button>
                <button
                  type="button"
                  @click="configTab = 'resizing'"
                  class="flex-1 py-1 text-[11px] font-medium rounded-md transition-all cursor-pointer text-center"
                  :class="
                    configTab === 'resizing'
                      ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-white shadow-xs font-semibold'
                      : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-200'
                  "
                >
                  Resizing
                </button>
                <button
                  type="button"
                  @click="configTab = 'preprocess'"
                  class="flex-1 py-1 text-[11px] font-medium rounded-md transition-all cursor-pointer text-center"
                  :class="
                    configTab === 'preprocess'
                      ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-white shadow-xs font-semibold'
                      : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-200'
                  "
                >
                  Preprocess
                </button>
              </div>
            </div>

            <!-- Popover Content Area -->
            <div class="p-3.5 space-y-3.5 overflow-y-auto">
              <!-- Tab 1: Detection & DBNet Postprocess -->
              <template v-if="configTab === 'detection'">
                <!-- Confidence Score -->
                <div class="space-y-1">
                  <div class="flex justify-between items-center">
                    <label class="font-medium text-neutral-700 dark:text-neutral-300">
                      Min Confidence Score (textScore)
                    </label>
                    <span class="font-mono text-[11px] text-primary font-semibold">
                      {{ Math.round(ocrConfig.textScore * 100) }}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    v-model.number="ocrConfig.textScore"
                    class="w-full accent-primary cursor-pointer"
                  />
                </div>

                <!-- Binarization Threshold -->
                <div class="space-y-1">
                  <div class="flex justify-between items-center">
                    <label class="font-medium text-neutral-700 dark:text-neutral-300">
                      DBNet Pixel Threshold (threshold)
                    </label>
                    <span class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold">
                      {{ ocrConfig.postprocessThreshold.toFixed(2) }}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    v-model.number="ocrConfig.postprocessThreshold"
                    class="w-full accent-primary cursor-pointer"
                  />
                </div>

                <!-- Box Threshold -->
                <div class="space-y-1">
                  <div class="flex justify-between items-center">
                    <label class="font-medium text-neutral-700 dark:text-neutral-300">
                      Box Confidence (boxThreshold)
                    </label>
                    <span class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold">
                      {{ ocrConfig.postprocessBoxThreshold.toFixed(2) }}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    v-model.number="ocrConfig.postprocessBoxThreshold"
                    class="w-full accent-primary cursor-pointer"
                  />
                </div>

                <!-- Unclip Ratio -->
                <div class="space-y-1">
                  <div class="flex justify-between items-center">
                    <label class="font-medium text-neutral-700 dark:text-neutral-300">
                      Bounding Box Expansion (unclipRatio)
                    </label>
                    <span class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold">
                      {{ ocrConfig.postprocessUnclipRatio.toFixed(1) }}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="3"
                    step="0.1"
                    v-model.number="ocrConfig.postprocessUnclipRatio"
                    class="w-full accent-primary cursor-pointer"
                  />
                </div>

                <!-- Max Candidates & Limit Side -->
                <div class="grid grid-cols-2 gap-2 pt-1 border-t border-neutral-100 dark:border-neutral-800">
                  <div class="space-y-1">
                    <label class="text-[11px] font-medium text-neutral-600 dark:text-neutral-400">
                      Max Candidates
                    </label>
                    <input
                      type="number"
                      min="100"
                      max="3000"
                      step="100"
                      v-model.number="ocrConfig.postprocessMaxCandidates"
                      class="w-full px-2 py-1 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                    />
                  </div>
                  <div class="space-y-1">
                    <label class="text-[11px] font-medium text-neutral-600 dark:text-neutral-400">
                      Limit Side Length
                    </label>
                    <input
                      type="number"
                      min="320"
                      max="3840"
                      step="160"
                      v-model.number="ocrConfig.detectionLimitSideLen"
                      class="w-full px-2 py-1 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                    />
                  </div>
                </div>

                <div class="flex items-center justify-between pt-1">
                  <div class="flex items-center gap-2">
                    <label class="text-[11px] text-neutral-600 dark:text-neutral-400">Limit Type:</label>
                    <select
                      v-model="ocrConfig.detectionLimitType"
                      class="px-2 py-0.5 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                    >
                      <option value="max">Max</option>
                      <option value="min">Min</option>
                    </select>
                  </div>
                  <label class="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      v-model="ocrConfig.postprocessUseDilation"
                      class="rounded accent-primary w-3.5 h-3.5 cursor-pointer"
                    />
                    <span class="text-neutral-700 dark:text-neutral-300 text-[11px]">Use Dilation</span>
                  </label>
                </div>
              </template>

              <!-- Tab 2: Layout & Pipeline Stages -->
              <template v-else-if="configTab === 'layout'">
                <!-- lineYThreshold -->
                <div class="space-y-1">
                  <div class="flex justify-between items-center">
                    <label class="font-medium text-neutral-700 dark:text-neutral-300">
                      Row Center Tolerance (lineYThreshold)
                    </label>
                    <span class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold">
                      {{ ocrConfig.lineYThreshold.toFixed(2) }}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    v-model.number="ocrConfig.lineYThreshold"
                    class="w-full accent-primary cursor-pointer"
                  />
                  <p class="text-[10px] text-neutral-400">Vertical center distance tolerance for same-row clustering.</p>
                </div>

                <!-- wordXThreshold -->
                <div class="space-y-1">
                  <div class="flex justify-between items-center">
                    <label class="font-medium text-neutral-700 dark:text-neutral-300">
                      Word Gap Tolerance (wordXThreshold)
                    </label>
                    <span class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold">
                      {{ ocrConfig.wordXThreshold.toFixed(2) }}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    v-model.number="ocrConfig.wordXThreshold"
                    class="w-full accent-primary cursor-pointer"
                  />
                  <p class="text-[10px] text-neutral-400">Horizontal gap multiplier when reconstructing words.</p>
                </div>

                <!-- Classification & Orientation Toggles -->
                <div class="space-y-2 pt-2 border-t border-neutral-100 dark:border-neutral-800">
                  <label class="flex items-center justify-between cursor-pointer">
                    <span class="text-neutral-700 dark:text-neutral-300">
                      Orientation Classifier (0° / 180°)
                    </span>
                    <input
                      type="checkbox"
                      v-model="ocrConfig.classification"
                      class="rounded accent-primary w-4 h-4 cursor-pointer"
                    />
                  </label>
                  <label class="flex items-center justify-between cursor-pointer">
                    <span class="text-neutral-700 dark:text-neutral-300">
                      4-Way Direction Correction (orientation)
                    </span>
                    <input
                      type="checkbox"
                      v-model="ocrConfig.orientation"
                      class="rounded accent-primary w-4 h-4 cursor-pointer"
                    />
                  </label>
                </div>
              </template>

              <!-- Tab 3: Image Resizing & Geometry -->
              <template v-else-if="configTab === 'resizing'">
                <div class="space-y-3">
                  <label class="flex items-center justify-between cursor-pointer pb-2 border-b border-neutral-100 dark:border-neutral-800">
                    <div>
                      <span class="font-medium text-neutral-800 dark:text-neutral-200">
                        Enable Geometry Resize Overrides
                      </span>
                      <p class="text-[10px] text-neutral-400">Override input dimensions before sending to detector.</p>
                    </div>
                    <input
                      type="checkbox"
                      v-model="ocrConfig.enableResize"
                      class="rounded accent-primary w-4 h-4 cursor-pointer"
                    />
                  </label>

                  <div
                    class="space-y-2.5 transition-opacity"
                    :class="{ 'opacity-40 pointer-events-none': !ocrConfig.enableResize }"
                  >
                    <div class="grid grid-cols-2 gap-2">
                      <div class="space-y-1">
                        <label class="text-[11px] font-medium text-neutral-600 dark:text-neutral-400">
                          Max Side Length (px)
                        </label>
                        <input
                          type="number"
                          min="320"
                          max="4096"
                          step="128"
                          v-model.number="ocrConfig.maxSideLen"
                          class="w-full px-2 py-1 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                        />
                      </div>
                      <div class="space-y-1">
                        <label class="text-[11px] font-medium text-neutral-600 dark:text-neutral-400">
                          Min Side Length (px)
                        </label>
                        <input
                          type="number"
                          min="100"
                          max="2048"
                          step="64"
                          v-model.number="ocrConfig.minSideLen"
                          class="w-full px-2 py-1 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                        />
                      </div>
                    </div>

                    <div class="grid grid-cols-2 gap-2">
                      <div class="space-y-1">
                        <label class="text-[11px] font-medium text-neutral-600 dark:text-neutral-400">
                          Min Height (px)
                        </label>
                        <input
                          type="number"
                          min="16"
                          max="512"
                          step="16"
                          v-model.number="ocrConfig.minHeight"
                          class="w-full px-2 py-1 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                        />
                      </div>
                      <div class="space-y-1">
                        <label class="text-[11px] font-medium text-neutral-600 dark:text-neutral-400">
                          Aspect Ratio (-1 for auto)
                        </label>
                        <input
                          type="number"
                          min="-1"
                          max="10"
                          step="0.1"
                          v-model.number="ocrConfig.widthHeightRatio"
                          class="w-full px-2 py-1 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </template>

              <!-- Tab 4: Preprocess & Optimization -->
              <template v-else-if="configTab === 'preprocess'">
                <div class="space-y-3">
                  <div class="space-y-2">
                    <label class="flex items-center justify-between cursor-pointer">
                      <span class="text-neutral-700 dark:text-neutral-300">
                        Adaptive Lighting Calibration
                      </span>
                      <input
                        type="checkbox"
                        v-model="ocrConfig.calibrationEnabled"
                        class="rounded accent-primary w-4 h-4 cursor-pointer"
                      />
                    </label>

                    <label class="flex items-center justify-between cursor-pointer">
                      <span class="text-neutral-700 dark:text-neutral-300">
                        Descreen Pattern Backgrounds (Guilloche)
                      </span>
                      <input
                        type="checkbox"
                        v-model="ocrConfig.descreenBackground"
                        class="rounded accent-primary w-4 h-4 cursor-pointer"
                      />
                    </label>

                    <!-- Descreen Strength -->
                    <div class="space-y-1 pt-1">
                      <div class="flex justify-between items-center">
                        <label class="text-[11px] text-neutral-600 dark:text-neutral-400">
                          Descreen Strength
                        </label>
                        <span class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold">
                          {{ ocrConfig.descreenStrength.toFixed(2) }}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="1"
                        step="0.05"
                        v-model.number="ocrConfig.descreenStrength"
                        class="w-full accent-primary cursor-pointer"
                      />
                    </div>
                  </div>

                  <div class="space-y-2 pt-2 border-t border-neutral-100 dark:border-neutral-800">
                    <label class="flex items-center justify-between cursor-pointer">
                      <span class="text-neutral-700 dark:text-neutral-300">
                        SIMD & Fast Crop Optimization
                      </span>
                      <input
                        type="checkbox"
                        v-model="ocrConfig.optimizationEnabled"
                        class="rounded accent-primary w-4 h-4 cursor-pointer"
                      />
                    </label>

                    <div class="grid grid-cols-2 gap-2 pt-1">
                      <div class="space-y-1">
                        <label class="text-[11px] text-neutral-600 dark:text-neutral-400">
                          Target Max Side (px)
                        </label>
                        <input
                          type="number"
                          min="640"
                          max="2560"
                          step="128"
                          v-model.number="ocrConfig.optimizationTargetMaxSide"
                          class="w-full px-2 py-1 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                        />
                      </div>
                      <div class="space-y-1">
                        <label class="text-[11px] text-neutral-600 dark:text-neutral-400">
                          Crop Padding (px)
                        </label>
                        <input
                          type="number"
                          min="0"
                          max="50"
                          step="2"
                          v-model.number="ocrConfig.cropPaddingX"
                          class="w-full px-2 py-1 text-xs rounded border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </div>
          </div>
        </div>

        <!-- Run OCR Button -->
        <button
          type="button"
          :disabled="status === 'initializing' || isScanning"
          @click="runOcr"
          class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-primary text-white hover:opacity-90 disabled:opacity-50 transition-all cursor-pointer shadow-sm"
        >
          <span
            v-if="isScanning || status === 'initializing'"
            class="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"
          />
          {{ isScanning ? 'Processing...' : status === 'initializing' ? 'Downloading...' : 'Run OCR' }}
        </button>
      </div>
    </div>

    <!-- Active Progress Bar (Download or OCR) -->
    <div
      v-if="progress.active"
      class="px-4 py-2 border-b border-primary/20 bg-primary/5 dark:bg-primary/10 transition-all"
    >
      <div class="flex items-center justify-between text-xs mb-1">
        <span class="font-medium text-primary flex items-center gap-2">
          <span
            class="inline-block w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin"
          />
          {{ progress.stageText }}
        </span>
        <span class="font-mono text-[11px] text-neutral-600 dark:text-neutral-400">
          {{ progress.detailText }}
        </span>
      </div>
      <div class="w-full bg-neutral-200 dark:bg-neutral-800 h-1.5 rounded-full overflow-hidden">
        <div
          class="bg-primary h-full rounded-full transition-all duration-200"
          :style="{ width: `${progress.percent}%` }"
        />
      </div>
    </div>

    <!-- Dedicated Row: Quick Sample Selector (1 full row, not flexed with model picker) -->
    <div
      class="p-3.5 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-100/50 dark:bg-neutral-900/50 flex items-center gap-3 overflow-x-auto"
    >
      <span
        class="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 shrink-0"
      >
        Sample Image:
      </span>
      <button
        v-for="sample in samples"
        :key="sample.id"
        type="button"
        @click="pickSample(sample)"
        class="px-3 py-1 text-xs rounded-lg border transition-all cursor-pointer shrink-0"
        :class="
          selectedSample === sample.id
            ? 'border-primary bg-primary/10 text-primary font-semibold'
            : 'border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 text-neutral-700 dark:text-neutral-300 bg-white dark:bg-neutral-900'
        "
      >
        {{ sample.name }}
      </button>

      <label
        class="px-2.5 py-1 text-xs rounded-lg border border-dashed border-neutral-300 dark:border-neutral-700 hover:border-primary text-neutral-600 dark:text-neutral-400 hover:text-primary transition-all cursor-pointer inline-flex items-center gap-1.5 bg-white dark:bg-neutral-900 shrink-0"
      >
        <span>+ Upload File</span>
        <input type="file" accept="image/*" class="hidden" @change="handleFileUpload" />
      </label>
    </div>

    <!-- Main Workspace: Image Visualizer + Inspector Panel -->
    <div class="grid grid-cols-1 lg:grid-cols-12 min-h-[480px]">
      <!-- Left: Interactive Image Canvas & Overlays (7 cols) -->
      <div
        class="lg:col-span-7 p-4 flex flex-col items-center justify-center bg-neutral-950/5 dark:bg-black/30 border-b lg:border-b-0 lg:border-r border-neutral-200 dark:border-neutral-800 overflow-auto"
      >
        <div
          class="relative max-w-full inline-block rounded-lg shadow-md overflow-hidden bg-white dark:bg-neutral-950"
        >
          <img
            ref="imgRef"
            :src="imageSrc"
            alt="OCR Target Document"
            @load="onImageLoad"
            class="block max-w-full max-h-[500px] w-auto h-auto object-contain"
          />

          <!-- Scanning Laser Bar Effect -->
          <div
            v-if="isScanning"
            class="absolute inset-0 pointer-events-none overflow-hidden bg-primary/5"
          >
            <div class="scanner-laser" />
          </div>

          <!-- Bounding Boxes Overlay -->
          <div
            v-if="!isScanning && results.length > 0"
            class="absolute inset-0 pointer-events-none"
          >
            <div
              v-for="(item, idx) in results"
              :key="idx"
              class="absolute pointer-events-auto transition-all cursor-pointer"
              :style="{
                left: `${item.frame.left * scaleX}px`,
                top: `${item.frame.top * scaleY}px`,
                width: `${item.frame.width * scaleX}px`,
                height: `${item.frame.height * scaleY}px`,
                border:
                  hoveredIndex === idx
                    ? '2px solid #f97316'
                    : '1.5px solid rgba(59, 130, 246, 0.7)',
                backgroundColor:
                  hoveredIndex === idx ? 'rgba(249, 115, 22, 0.25)' : 'rgba(59, 130, 246, 0.15)',
              }"
              @mouseenter="hoveredIndex = idx"
              @mouseleave="hoveredIndex = null"
              @click="copyToClipboard(item.text)"
              :title="`${item.text} (${Math.round(item.score * 100)}%) - Click to copy`"
            />
          </div>
        </div>

        <p class="mt-3 text-[11px] text-neutral-400">
          Tip: Hover over bounding boxes to highlight recognized text. Click to copy.
        </p>
      </div>

      <!-- Right: Inspector & Output Panel (5 cols) -->
      <div class="lg:col-span-5 flex flex-col bg-white dark:bg-neutral-900">
        <!-- Granularity & Tab Header -->
        <div
          class="p-3 border-b border-neutral-200 dark:border-neutral-800 flex items-center justify-between gap-2"
        >
          <div class="flex items-center gap-1">
            <button
              v-for="mode in ['lines', 'words', 'spatial'] as OutputGranularity[]"
              :key="mode"
              type="button"
              @click="setGranularity(mode)"
              class="px-2.5 py-1 text-[11px] rounded capitalize transition-all cursor-pointer font-medium"
              :class="
                outputMode === mode
                  ? 'bg-primary text-white'
                  : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800'
              "
            >
              {{ mode }}
            </button>
          </div>

          <button
            type="button"
            @click="
              copyToClipboard(
                activeTab === 'spatial'
                  ? spatialText
                  : activeTab === 'csv'
                    ? csvExport
                    : activeTab === 'json'
                      ? jsonExport
                      : results.map((r) => r.text).join('\n')
              )
            "
            class="px-2.5 py-1 text-[11px] rounded border border-neutral-200 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-all cursor-pointer"
          >
            {{ copied ? '✓ Copied' : 'Copy All' }}
          </button>
        </div>

        <!-- Secondary Tabs: Structured List / Spatial / CSV / JSON -->
        <div
          class="flex border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50 text-[11px]"
        >
          <button
            type="button"
            @click="activeTab = 'structured'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer"
            :class="
              activeTab === 'structured'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            Items ({{ isScanning ? '...' : results.length }})
          </button>
          <button
            type="button"
            @click="activeTab = 'spatial'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer"
            :class="
              activeTab === 'spatial'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            Spatial Layout
          </button>
          <button
            type="button"
            @click="activeTab = 'csv'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer"
            :class="
              activeTab === 'csv'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            CSV
          </button>
          <button
            type="button"
            @click="activeTab = 'json'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer"
            :class="
              activeTab === 'json'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            JSON
          </button>
        </div>

        <!-- Tab Body Content -->
        <div class="flex-1 p-3 overflow-y-auto max-h-[420px] text-xs font-mono">
          <!-- Scanning Skeleton State -->
          <template v-if="isScanning">
            <!-- Structured items skeleton -->
            <div v-if="activeTab === 'structured'" class="space-y-2">
              <div
                v-for="i in 5"
                :key="i"
                class="p-2.5 rounded border border-neutral-200/70 dark:border-neutral-800/70 bg-neutral-100/60 dark:bg-neutral-900/60 flex items-center justify-between gap-3 animate-pulse"
              >
                <div
                  class="h-3.5 rounded bg-neutral-200 dark:bg-neutral-800"
                  :style="{ width: `${40 + ((i * 13) % 45)}%` }"
                />
                <div class="h-4 w-9 rounded bg-neutral-200 dark:bg-neutral-800 shrink-0" />
              </div>
            </div>

            <!-- Spatial ASCII layout skeleton -->
            <div
              v-else-if="activeTab === 'spatial'"
              class="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 space-y-3 animate-pulse"
            >
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-2/5" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-3/5 ml-12" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-4/5 ml-4" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-1/2" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-3/4 ml-8" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-2/3 ml-16" />
            </div>

            <!-- Raw Tabular / JSON / CSV skeleton -->
            <div
              v-else
              class="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 space-y-2.5 animate-pulse"
            >
              <div
                v-for="i in 6"
                :key="i"
                class="h-3 rounded bg-neutral-200 dark:bg-neutral-800"
                :style="{ width: `${30 + ((i * 17) % 65)}%` }"
              />
            </div>
          </template>

          <!-- Resolved OCR Output -->
          <template v-else>
            <!-- Structured list view -->
            <div v-if="activeTab === 'structured'" class="space-y-1.5">
              <div
                v-for="(item, idx) in results"
                :key="idx"
                @mouseenter="hoveredIndex = idx"
                @mouseleave="hoveredIndex = null"
                class="p-2 rounded border transition-all flex items-start justify-between gap-2"
                :class="
                  hoveredIndex === idx
                    ? 'border-primary bg-primary/5 dark:bg-primary/10'
                    : 'border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-900/50'
                "
              >
                <div class="flex-1 font-sans text-neutral-800 dark:text-neutral-200 select-all">
                  {{ item.text }}
                </div>
                <span
                  class="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold"
                  :class="
                    item.score >= 0.9
                      ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-400'
                      : 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-400'
                  "
                >
                  {{ Math.round(item.score * 100) }}%
                </span>
              </div>
              <p v-if="results.length === 0" class="text-neutral-400 p-4 text-center">
                No text detected or score below threshold.
              </p>
            </div>

            <!-- Spatial ASCII view -->
            <pre
              v-else-if="activeTab === 'spatial'"
              class="whitespace-pre overflow-x-auto text-neutral-800 dark:text-neutral-200 p-3 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 font-mono text-xs leading-relaxed select-text shadow-inner"
              style="
                font-family:
                  ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono',
                  'Courier New', monospace;
                tab-size: 4;
              "
              >{{
                spatialText || 'Run detection to see reconstructed 2D spatial text layout.'
              }}</pre>

            <!-- CSV view -->
            <pre
              v-else-if="activeTab === 'csv'"
              class="whitespace-pre overflow-x-auto text-neutral-700 dark:text-neutral-300 p-3 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 font-mono text-xs leading-relaxed select-text"
              style="
                font-family:
                  ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono',
                  'Courier New', monospace;
                tab-size: 4;
              "
              >{{ csvExport }}</pre>

            <!-- JSON view -->
            <pre
              v-else-if="activeTab === 'json'"
              class="whitespace-pre overflow-x-auto text-neutral-700 dark:text-neutral-300 p-3 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 font-mono text-xs leading-relaxed select-text"
              style="
                font-family:
                  ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono',
                  'Courier New', monospace;
                tab-size: 4;
              "
              >{{ jsonExport }}</pre>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.rusto-demo-container {
  font-family: inherit;
}

@keyframes scanline {
  0% {
    top: 0%;
  }
  50% {
    top: 96%;
  }
  100% {
    top: 0%;
  }
}

.scanner-laser {
  position: absolute;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent 0%, rgba(59, 130, 246, 0.9) 50%, transparent 100%);
  box-shadow: 0 0 12px 2px rgba(59, 130, 246, 0.7);
  animation: scanline 2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}
</style>
