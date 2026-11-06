<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue';
import { useRuntimeConfig } from '#app';
import {
  availableModels,
  availableLayoutModels,
  defaultModel,
  modelCategories,
  type OcrModelOption,
  type LayoutModelOption,
} from '../../types/ocrModels';
import { ocrProfilePresets } from '../../types/ocrPresets';
import { getSampleImages } from '../../types/ocrSamples';
import { useWebOcrSettings } from '../../composables/useWebOcrSettings';
import { useWebOcrExecution } from '../../composables/useWebOcrExecution';
import {
  fetchHuggingFaceModelIndex,
  getModelCandidateUrls,
  HUGGINGFACE_REPO,
  HUGGINGFACE_BASE,
} from '../../utils/ocrReleaseModels';
import {
  formatResultsTsv,
  buildActiveConfigJson,
  isRunConfigModified,
  getImageDimensions,
} from '../../utils/ocrResultFormatter';
import OcrActiveConfigView from '../OcrActiveConfigView.vue';
import OcrLoadingOverlay from '../OcrLoadingOverlay.vue';

const rawBaseURL = useRuntimeConfig().app.baseURL || '/';
const baseURL = rawBaseURL.endsWith('/') ? rawBaseURL : `${rawBaseURL}/`;

const samples = getSampleImages(baseURL);
const defaultSample = samples[0]!;

const {
  ocrConfig,
  isConfigOpen,
  configTab,
  toggleConfigPopover,
  resetConfigDefaults,
  applyPreset,
  isPresetActive,
} = useWebOcrSettings();

const selectedModel = ref<string>('indraocr');
const currentModel = computed<OcrModelOption>(
  () => availableModels.find((m) => m.id === selectedModel.value) ?? defaultModel
);
const selectedLayoutModelMeta = computed<LayoutModelOption>(() => {
  return (
    availableLayoutModels.find((m) => m.id === ocrConfig.value.layoutModel) ||
    availableLayoutModels[0]!
  );
});

function onLayoutModelChange() {
  if (ocrConfig.value.layoutModel === 'none') {
    ocrConfig.value.enableReadingOrder = false;
  } else {
    ocrConfig.value.enableReadingOrder = true;
  }
  resetResults();
}

const selectedSample = ref<string>('invoice');
const imageSrc = ref<string>(defaultSample.url);
const outputMode = ref<'lines' | 'words'>('lines');
const activeTab = ref<'structured' | 'spatial' | 'tsv' | 'json' | 'config'>('structured');
const tsvViewMode = ref<'table' | 'raw'>('table');
const copied = ref<boolean>(false);
const isWebGpuSupported = ref<boolean>(false);

const {
  status,
  statusMessage,
  isScanning,
  progress,
  durationMs,
  results,
  spatialText,
  activeRunConfig,
  ocrError,
  hoveredIndex,
  initEngine,
  switchModel,
  resetResults,
  runOcr,
} = useWebOcrExecution({
  baseURL,
  imageSrc,
  ocrConfig,
  selectedModel,
  currentModel,
  outputMode,
});

const isConfigModifiedSinceRun = computed(() =>
  isRunConfigModified(activeRunConfig.value, selectedModel.value, outputMode.value, ocrConfig.value)
);

const activeScorePercent = computed(() => {
  if (!activeRunConfig.value) return 50;
  const score =
    activeRunConfig.value.options.postRecognition?.scoreThreshold ??
    activeRunConfig.value.rawConfig.textScore ??
    0.5;
  return Math.round(score * 100);
});

const imgRef = ref<HTMLImageElement | null>(null);
const imgDims = ref({ naturalW: 1, naturalH: 1, renderW: 1, renderH: 1 });
const scaleX = computed(() => imgDims.value.renderW / imgDims.value.naturalW || 1);
const scaleY = computed(() => imgDims.value.renderH / imgDims.value.naturalH || 1);

function setGranularity(mode: 'lines' | 'words') {
  outputMode.value = mode;
  if (results.value.length > 0 || spatialText.value) runOcr();
}

function pickSample(sample: (typeof samples)[0]) {
  resetResults();
  selectedSample.value = sample.id;
  imageSrc.value = sample.url;
}

function handleFileUpload(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  resetResults();
  selectedSample.value = 'custom';
  imageSrc.value = `${URL.createObjectURL(file)}#${encodeURIComponent(file.name)}`;
}

function updateImageDimensions() {
  if (!imgRef.value) return;
  const img = imgRef.value;
  const rect = img.getBoundingClientRect();
  const rW = rect.width > 0 ? rect.width : img.clientWidth || 0;
  const rH = rect.height > 0 ? rect.height : img.clientHeight || 0;
  if (img.naturalWidth > 0 && img.naturalHeight > 0 && rW > 0 && rH > 0) {
    imgDims.value = {
      naturalW: img.naturalWidth,
      naturalH: img.naturalHeight,
      renderW: rW,
      renderH: rH,
    };
  }
}

function onImageLoad() {
  nextTick(() => updateImageDimensions());
}

watch(results, () => {
  nextTick(() => updateImageDimensions());
});

let resizeObserver: ResizeObserver | null = null;

async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    copied.value = true;
    setTimeout(() => (copied.value = false), 2000);
  } catch (err) {
    console.error('Copy failed:', err);
  }
}

const tsvExport = computed(() => formatResultsTsv(results.value));
const jsonExport = computed(() => JSON.stringify(results.value, null, 2));
const activeConfigJson = computed(() =>
  buildActiveConfigJson(
    activeRunConfig.value,
    currentModel.value.name,
    outputMode.value,
    ocrConfig.value
  )
);

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && isConfigOpen.value) {
    isConfigOpen.value = false;
  }
}

onMounted(() => {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator && !!navigator.gpu) {
    isWebGpuSupported.value = true;
  }
  status.value = 'idle';
  statusMessage.value = `${currentModel.value.name} selected (click Run OCR to load)`;
  fetchHuggingFaceModelIndex().catch(() => {});
  nextTick(() => updateImageDimensions());
  if (typeof ResizeObserver !== 'undefined' && imgRef.value) {
    resizeObserver = new ResizeObserver(() => updateImageDimensions());
    resizeObserver.observe(imgRef.value);
  }
  if (typeof window !== 'undefined') {
    window.addEventListener('resize', updateImageDimensions);
    window.addEventListener('keydown', onKeydown);
  }
});

onUnmounted(() => {
  resizeObserver?.disconnect();
  if (typeof window !== 'undefined') {
    window.removeEventListener('resize', updateImageDimensions);
    window.removeEventListener('keydown', onKeydown);
  }
});
</script>

<template>
  <div
    class="rusto-demo-container not-prose relative my-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 shadow-xl overflow-hidden"
  >
    <!-- Blocking Whole-Component Loading Overlay with Circular Progress -->
    <OcrLoadingOverlay :progress="progress" />

    <!-- Top Action & Status Bar -->
    <div
      class="demo-header relative flex flex-wrap items-center justify-between gap-3 p-3.5 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950"
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

        <!-- Consolidated Model Picker Select with Category Optgroups -->
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
              class="appearance-none pl-2.5 pr-7 py-1 text-xs font-semibold rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-neutral-800 dark:text-neutral-200 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary disabled:opacity-50 disabled:cursor-not-allowed shadow-xs transition-all enabled:cursor-pointer"
            >
              <optgroup v-for="cat in modelCategories" :key="cat.id" :label="cat.name">
                <option
                  v-for="model in availableModels.filter((m) => m.category === cat.id)"
                  :key="model.id"
                  :value="model.id"
                >
                  {{ model.name }} ({{ model.size }})
                </option>
              </optgroup>
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

          <!-- Hugging Face Source Link Pill -->
          <a
            :href="`https://huggingface.co/${HUGGINGFACE_REPO}/tree/main`"
            target="_blank"
            rel="noopener noreferrer"
            class="hidden md:inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full font-medium bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-amber-700 dark:text-amber-300 hover:opacity-80 transition-opacity"
            title="Model weights hosted on Hugging Face Hub (byrizki/rten-models)"
          >
            <span>HF Hub</span>
            <svg class="w-3 h-3 text-amber-500" viewBox="0 0 20 20" fill="currentColor">
              <path
                fill-rule="evenodd"
                d="M4.25 5.5a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 0 0 .75-.75v-4a.75.75 0 0 1 1.5 0v4A2.25 2.25 0 0 1 12.75 17h-8.5A2.25 2.25 0 0 1 2 14.75v-8.5A2.25 2.25 0 0 1 4.25 4h4a.75.75 0 0 1 0 1.5h-4Zm6.75-2a.75.75 0 0 1 .75-.75h5.5c.414 0 .75.336.75.75v5.5a.75.75 0 0 1-1.5 0V4.56l-6.72 6.72a.75.75 0 1 1-1.06-1.06l6.72-6.72h-4.44a.75.75 0 0 1-.75-.75Z"
                clip-rule="evenodd"
              />
            </svg>
          </a>

          <!-- Hardware Acceleration Indicator Pill -->
          <button
            type="button"
            @click="
              configTab = 'hardware';
              isConfigOpen = true;
            "
            class="hidden sm:inline-flex items-center gap-1.5 text-[11px] px-2.5 py-0.5 rounded-full font-medium border transition-colors cursor-pointer"
            :class="
              ocrConfig.useWebGpu
                ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-300 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300'
                : 'bg-neutral-100 dark:bg-neutral-800 border-neutral-200 dark:border-neutral-700 text-neutral-600 dark:text-neutral-400'
            "
            :title="
              ocrConfig.useWebGpu
                ? 'WebGPU Acceleration Active (Click to configure)'
                : 'WASM SIMD128 Active (Click to configure WebGPU)'
            "
          >
            <span
              class="w-1.5 h-1.5 rounded-full"
              :class="ocrConfig.useWebGpu ? 'bg-emerald-500 animate-pulse' : 'bg-neutral-400'"
            />
            <span>{{ ocrConfig.useWebGpu ? 'WebGPU' : 'WASM SIMD' }}</span>
          </button>
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
        <button
          type="button"
          @click="toggleConfigPopover"
          class="relative inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-all cursor-pointer shadow-xs"
          :class="{ 'border-primary text-primary z-50': isConfigOpen }"
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
          {{
            isScanning ? 'Processing...' : status === 'initializing' ? 'Downloading...' : 'Run OCR'
          }}
        </button>
      </div>

      <!-- Backdrop for closing popover when clicking outside -->
      <div
        v-if="isConfigOpen"
        class="fixed inset-0 z-40 bg-black/10 dark:bg-black/30 sm:bg-transparent"
        @click="isConfigOpen = false"
      />

      <!-- Floating Popover Modal (anchored to demo-header) -->
      <div
        v-if="isConfigOpen"
        class="absolute right-3.5 top-full mt-1.5 w-[calc(100%-1.75rem)] sm:w-[500px] max-w-[calc(100%-1.75rem)] max-h-[82vh] flex flex-col rounded-xl shadow-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 z-50 text-xs overflow-hidden"
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

        <!-- Presets Bar -->
        <div
          class="px-3.5 py-2 border-b border-neutral-100 dark:border-neutral-800 bg-neutral-50/80 dark:bg-neutral-800/40 shrink-0"
        >
          <div class="flex items-center gap-1.5 flex-wrap">
            <span
              class="text-[10px] text-neutral-400 font-semibold uppercase tracking-wider shrink-0 mr-0.5"
            >
              Preset:
            </span>
            <button
              v-for="preset in ocrProfilePresets"
              :key="preset.id"
              type="button"
              @click="applyPreset(preset)"
              class="px-2 py-0.5 text-[10px] rounded-md transition-all border cursor-pointer"
              :class="
                isPresetActive(preset)
                  ? 'bg-primary text-white border-primary font-semibold shadow-xs'
                  : 'border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 text-neutral-600 dark:text-neutral-300 hover:border-neutral-400 dark:hover:border-neutral-600'
              "
              :title="preset.description"
            >
              {{ preset.name }}
            </button>
          </div>
        </div>

        <!-- Main Content Area: Left Vertical Icon Rail + Right Settings Panel -->
        <div class="flex flex-1 min-h-0 overflow-hidden">
          <!-- Left Vertical Icon Rail -->
          <div
            class="w-12 shrink-0 border-r border-neutral-100 dark:border-neutral-800 p-1.5 flex flex-col gap-1.5 bg-neutral-50/80 dark:bg-neutral-950/40 items-center select-none"
          >
            <!-- 1. Detection -->
            <button
              type="button"
              @click="configTab = 'detection'"
              class="w-9 h-9 flex items-center justify-center rounded-lg transition-all cursor-pointer relative group"
              :class="
                configTab === 'detection'
                  ? 'bg-primary text-white shadow-xs'
                  : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-200/60 dark:hover:bg-neutral-800'
              "
              title="Detection & DBNet Thresholds"
            >
              <svg
                class="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <path d="M3 7V5a2 2 0 0 1 2-2h2" />
                <path d="M17 3h2a2 2 0 0 1 2 2v2" />
                <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
                <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </button>

            <!-- 2. Recognition -->
            <button
              type="button"
              @click="configTab = 'recognition'"
              class="w-9 h-9 flex items-center justify-center rounded-lg transition-all cursor-pointer relative group"
              :class="
                configTab === 'recognition'
                  ? 'bg-primary text-white shadow-xs'
                  : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-200/60 dark:hover:bg-neutral-800'
              "
              title="Recognition & Batch Size"
            >
              <svg
                class="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <polyline points="4 7 4 4 20 4 20 7" />
                <line x1="9" y1="20" x2="15" y2="20" />
                <line x1="12" y1="4" x2="12" y2="20" />
              </svg>
            </button>

            <!-- 3. Reading & Layout -->
            <button
              type="button"
              @click="configTab = 'layout'"
              class="w-9 h-9 flex items-center justify-center rounded-lg transition-all cursor-pointer relative group"
              :class="
                configTab === 'layout'
                  ? 'bg-primary text-white shadow-xs'
                  : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-200/60 dark:hover:bg-neutral-800'
              "
              title="Reading Order & Layout Model"
            >
              <svg
                class="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <rect x="3" y="3" width="7" height="18" rx="1" />
                <rect x="14" y="3" width="7" height="18" rx="1" />
              </svg>
            </button>

            <!-- 4. Resizing -->
            <button
              type="button"
              @click="configTab = 'resizing'"
              class="w-9 h-9 flex items-center justify-center rounded-lg transition-all cursor-pointer relative group"
              :class="
                configTab === 'resizing'
                  ? 'bg-primary text-white shadow-xs'
                  : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-200/60 dark:hover:bg-neutral-800'
              "
              title="Image Resizing & Constraints"
            >
              <svg
                class="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <polyline points="15 3 21 3 21 9" />
                <polyline points="9 21 3 21 3 15" />
                <line x1="21" y1="3" x2="14" y2="10" />
                <line x1="3" y1="21" x2="10" y2="14" />
              </svg>
            </button>

            <!-- 5. Preprocess -->
            <button
              type="button"
              @click="configTab = 'preprocess'"
              class="w-9 h-9 flex items-center justify-center rounded-lg transition-all cursor-pointer relative group"
              :class="
                configTab === 'preprocess'
                  ? 'bg-primary text-white shadow-xs'
                  : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-200/60 dark:hover:bg-neutral-800'
              "
              title="Preprocess & Calibration"
            >
              <svg
                class="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <line x1="4" y1="21" x2="4" y2="14" />
                <line x1="4" y1="10" x2="4" y2="3" />
                <line x1="12" y1="21" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12" y2="3" />
                <line x1="20" y1="21" x2="20" y2="16" />
                <line x1="20" y1="12" x2="20" y2="3" />
                <line x1="1" y1="14" x2="7" y2="14" />
                <line x1="9" y1="8" x2="15" y2="8" />
                <line x1="17" y1="16" x2="23" y2="16" />
              </svg>
            </button>

            <!-- 6. Hardware Acceleration -->
            <button
              type="button"
              @click="configTab = 'hardware'"
              class="w-9 h-9 flex items-center justify-center rounded-lg transition-all cursor-pointer relative group"
              :class="
                configTab === 'hardware'
                  ? 'bg-primary text-white shadow-xs'
                  : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-200/60 dark:hover:bg-neutral-800'
              "
              title="Hardware Acceleration (WebGPU / SIMD)"
            >
              <svg
                class="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <rect x="4" y="4" width="16" height="16" rx="2" />
                <rect x="9" y="9" width="6" height="6" />
                <line x1="9" y1="1" x2="9" y2="4" />
                <line x1="15" y1="1" x2="15" y2="4" />
                <line x1="9" y1="20" x2="9" y2="23" />
                <line x1="15" y1="20" x2="15" y2="23" />
                <line x1="20" y1="9" x2="23" y2="9" />
                <line x1="20" y1="14" x2="23" y2="14" />
                <line x1="1" y1="9" x2="4" y2="9" />
                <line x1="1" y1="14" x2="4" y2="14" />
              </svg>
            </button>

            <!-- 7. HF Models -->
            <button
              type="button"
              @click="configTab = 'models'"
              class="w-9 h-9 flex items-center justify-center rounded-lg transition-all cursor-pointer relative group"
              :class="
                configTab === 'models'
                  ? 'bg-primary text-white shadow-xs'
                  : 'text-neutral-500 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-200/60 dark:hover:bg-neutral-800'
              "
              title="Hugging Face Model Assets"
            >
              <svg
                class="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <path
                  d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"
                />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
            </button>
          </div>

          <!-- Right Settings Panel -->
          <div class="flex-1 min-w-0 p-3.5 space-y-3.5 overflow-y-auto max-h-[480px]">
            <!-- Active Tab Header -->
            <div
              class="flex items-center justify-between pb-1 border-b border-neutral-100 dark:border-neutral-800"
            >
              <span class="font-semibold text-xs text-neutral-800 dark:text-neutral-200">
                {{
                  configTab === 'detection'
                    ? 'Detection & DBNet Thresholds'
                    : configTab === 'recognition'
                      ? 'Recognition & Batching'
                      : configTab === 'layout'
                        ? 'Reading Order & Layout Model'
                        : configTab === 'resizing'
                          ? 'Image Resizing & Constraints'
                          : configTab === 'preprocess'
                            ? 'Preprocess & Calibration'
                            : configTab === 'hardware'
                              ? 'Hardware Acceleration (WebGPU / SIMD)'
                              : 'Hugging Face Model Assets'
                }}
              </span>
            </div>
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
                  <span
                    class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold"
                  >
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
                  <span
                    class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold"
                  >
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
                  <span
                    class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold"
                  >
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
              <div
                class="grid grid-cols-2 gap-2 pt-1 border-t border-neutral-100 dark:border-neutral-800"
              >
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
                  <label class="text-[11px] text-neutral-600 dark:text-neutral-400"
                    >Limit Type:</label
                  >
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
                  <span class="text-neutral-700 dark:text-neutral-300 text-[11px]"
                    >Use Dilation</span
                  >
                </label>
              </div>

              <!-- Quick Side Presets -->
              <div
                class="flex items-center gap-1.5 pt-1.5 border-t border-neutral-100 dark:border-neutral-800"
              >
                <span class="text-[10px] text-neutral-400">Quick limit:</span>
                <button
                  v-for="side in [640, 960, 1280, 1920]"
                  :key="side"
                  type="button"
                  @click="ocrConfig.detectionLimitSideLen = side"
                  class="px-1.5 py-0.5 text-[10px] rounded border transition-colors cursor-pointer"
                  :class="
                    ocrConfig.detectionLimitSideLen === side
                      ? 'bg-primary text-white border-primary font-semibold'
                      : 'border-neutral-200 dark:border-neutral-700 text-neutral-600 dark:text-neutral-400 hover:border-neutral-400'
                  "
                >
                  {{ side }}px
                </button>
              </div>
            </template>

            <!-- Tab 2: Recognition Engine Settings -->
            <template v-else-if="configTab === 'recognition'">
              <!-- Recognition Batch Size -->
              <div class="space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-neutral-700 dark:text-neutral-300">
                    Recognition Batch Size (rec_batch_num)
                  </label>
                  <span class="font-mono text-[11px] text-primary font-semibold">
                    {{ ocrConfig.recBatchNum }}
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="16"
                  step="1"
                  v-model.number="ocrConfig.recBatchNum"
                  class="w-full accent-primary cursor-pointer"
                />
                <p class="text-[10px] text-neutral-400">
                  {{
                    currentModel.preset === 'indra'
                      ? 'Indra Linenet: default batch size 1 eliminates padding overhead and gives ~28.5% faster recognition.'
                      : 'PP-OCR: recommended batch size 4–6 balances vectorization and memory.'
                  }}
                </p>
              </div>

              <!-- Text Score Cutoff -->
              <div class="space-y-1 pt-1.5 border-t border-neutral-100 dark:border-neutral-800">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-neutral-700 dark:text-neutral-300">
                    Min Text Confidence (textScore)
                  </label>
                  <span
                    class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold"
                  >
                    {{ Math.round(ocrConfig.textScore * 100) }}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.95"
                  step="0.05"
                  v-model.number="ocrConfig.textScore"
                  class="w-full accent-primary cursor-pointer"
                />
                <p class="text-[10px] text-neutral-400">
                  Filters out low-confidence OCR predictions from the final output.
                </p>
              </div>
            </template>

            <!-- Tab 3: Reading Order & Layout Geometry -->
            <template v-else-if="configTab === 'layout'">
              <!-- Layout & Reading Model Selection -->
              <div
                class="space-y-1.5 bg-neutral-50 dark:bg-neutral-800/40 p-2.5 rounded-lg border border-neutral-200/60 dark:border-neutral-700/60"
              >
                <div class="flex justify-between items-center">
                  <label class="font-medium text-neutral-800 dark:text-neutral-200 text-xs">
                    Layout / Structure Model
                  </label>
                  <span class="font-mono text-[10px] text-primary font-semibold">
                    {{ selectedLayoutModelMeta.size }}
                  </span>
                </div>
                <select
                  v-model="ocrConfig.layoutModel"
                  @change="onLayoutModelChange"
                  class="w-full text-xs font-medium rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 py-1.5 px-2.5 text-neutral-800 dark:text-neutral-200 focus:outline-none focus:ring-1 focus:ring-primary cursor-pointer"
                >
                  <optgroup label="Indra OCR (Transformer)">
                    <option value="indra">Indra Neural Reading Order (2.9 MB)</option>
                  </optgroup>
                  <optgroup label="PP-Structure / PaddleOCR">
                    <option value="cdla">PP-OCR CDLA Document Layout (8.4 MB)</option>
                    <option value="publaynet">PP-OCR PubLayNet Paper Layout (8.4 MB)</option>
                  </optgroup>
                  <optgroup label="Built-in / Baseline">
                    <option value="heuristic">Geometric Column Flow (0 KB - Built-in)</option>
                    <option value="none">None / Linear Raster Scan (Disabled)</option>
                  </optgroup>
                </select>
                <p class="text-[10px] text-neutral-500 dark:text-neutral-400 leading-tight">
                  {{ selectedLayoutModelMeta.description }}
                </p>
                <div
                  v-if="selectedLayoutModelMeta.downloadUrl"
                  class="pt-1 flex items-center justify-between text-[10px]"
                >
                  <span class="text-neutral-400">HF Weight:</span>
                  <a
                    :href="selectedLayoutModelMeta.downloadUrl"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="text-primary hover:underline font-mono truncate max-w-[220px]"
                  >
                    {{ selectedLayoutModelMeta.assetFile }}
                  </a>
                </div>
              </div>

              <!-- Reading Order Recovery -->
              <div
                class="space-y-1.5 bg-neutral-50 dark:bg-neutral-800/40 p-2.5 rounded-lg border border-neutral-200/60 dark:border-neutral-700/60"
              >
                <div class="flex items-center justify-between">
                  <div>
                    <span class="font-medium text-neutral-800 dark:text-neutral-200 text-xs">
                      Multi-Column Reading Order
                    </span>
                    <p class="text-[10px] text-neutral-500 dark:text-neutral-400">
                      enableReadingOrder (Natural Column Flow)
                    </p>
                  </div>
                  <label class="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      v-model="ocrConfig.enableReadingOrder"
                      class="sr-only peer"
                    />
                    <div
                      class="w-8 h-4 bg-neutral-300 peer-focus:outline-none rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all dark:border-neutral-600 peer-checked:bg-primary"
                    ></div>
                  </label>
                </div>
                <p class="text-[10px] text-neutral-500 dark:text-neutral-400 leading-tight">
                  Partitions dense multi-column layouts (newspapers, magazines, articles) into
                  natural human reading sequence (Column 1 top-to-bottom, then Column 2) instead of
                  horizontal scan.
                </p>
              </div>

              <!-- lineYThreshold -->
              <div class="space-y-1 pt-1.5">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-neutral-700 dark:text-neutral-300">
                    Row Center Tolerance (lineYThreshold)
                  </label>
                  <span
                    class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold"
                  >
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
                <p class="text-[10px] text-neutral-400">
                  Vertical center distance multiplier for same-line clustering.
                </p>
              </div>

              <!-- wordXThreshold -->
              <div class="space-y-1 pt-1 border-t border-neutral-100 dark:border-neutral-800">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-neutral-700 dark:text-neutral-300">
                    Word Gap Tolerance (wordXThreshold)
                  </label>
                  <span
                    class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold"
                  >
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
                <p class="text-[10px] text-neutral-400">
                  Horizontal gap multiplier when reconstructing words and columns.
                </p>
              </div>

              <!-- Pipeline Auxiliaries: Unwarp, Classification & Orientation -->
              <div class="space-y-2.5 pt-2 border-t border-neutral-100 dark:border-neutral-800">
                <label class="flex items-center justify-between cursor-pointer">
                  <div>
                    <span class="text-neutral-700 dark:text-neutral-300 font-medium">
                      Document Auto-Unwarp (withAutoUnwarp)
                    </span>
                    <p class="text-[10px] text-neutral-400">
                      3D mesh flattening for curved book pages and folded documents.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    v-model="ocrConfig.autoUnwarp"
                    class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
                  />
                </label>
                <label class="flex items-center justify-between cursor-pointer">
                  <div>
                    <span class="text-neutral-700 dark:text-neutral-300 font-medium">
                      Line Orientation Classifier (withLineClassification)
                    </span>
                    <p class="text-[10px] text-neutral-400">
                      Detects and rotates upside-down text lines (0° / 180°).
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    v-model="ocrConfig.lineClassification"
                    class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
                  />
                </label>
                <label class="flex items-center justify-between cursor-pointer">
                  <div>
                    <span class="text-neutral-700 dark:text-neutral-300 font-medium">
                      Page Orientation Detection (withOrientationDetection)
                    </span>
                    <p class="text-[10px] text-neutral-400">
                      Full-page 4-way rotation correction (0°, 90°, 180°, 270°).
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    v-model="ocrConfig.orientationDetection"
                    class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
                  />
                </label>
              </div>
            </template>

            <!-- Tab 3: Image Resizing & Geometry -->
            <template v-else-if="configTab === 'resizing'">
              <div class="space-y-3">
                <label
                  class="flex items-center justify-between cursor-pointer pb-2 border-b border-neutral-100 dark:border-neutral-800"
                >
                  <div>
                    <span class="font-medium text-neutral-800 dark:text-neutral-200">
                      Enable Geometry Resize Overrides
                    </span>
                    <p class="text-[10px] text-neutral-400">
                      Override input dimensions before sending to detector.
                    </p>
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
                      <span
                        class="font-mono text-[11px] text-neutral-600 dark:text-neutral-300 font-semibold"
                      >
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

            <!-- Tab 6: Hardware Acceleration (WebGPU / SIMD) -->
            <template v-else-if="configTab === 'hardware'">
              <!-- WebGPU Toggle Card -->
              <div
                class="space-y-2 bg-neutral-50 dark:bg-neutral-800/40 p-3 rounded-lg border border-neutral-200/60 dark:border-neutral-700/60"
              >
                <div class="flex items-center justify-between">
                  <div>
                    <span class="font-medium text-neutral-800 dark:text-neutral-200 text-xs">
                      WebGPU Hardware Acceleration
                    </span>
                    <p class="text-[10px] text-neutral-500 dark:text-neutral-400">
                      useWebGpu (Pluggable GPU Compute Runtime)
                    </p>
                  </div>
                  <label class="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" v-model="ocrConfig.useWebGpu" class="sr-only peer" />
                    <div
                      class="w-8 h-4 bg-neutral-300 peer-focus:outline-none rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all dark:border-neutral-600 peer-checked:bg-primary"
                    ></div>
                  </label>
                </div>
                <p class="text-[10px] text-neutral-500 dark:text-neutral-400 leading-tight">
                  Enables client-side GPU compute shaders for matrix multiplication (GEMM) and
                  convolutional layers via the pluggable
                  <code class="text-primary font-mono font-semibold">@rustors/webgpu</code> /
                  <code class="text-primary font-mono font-semibold">rusto-nn-webgpu</code> package.
                </p>
                <div
                  class="pt-1 flex items-center justify-between text-[10px] border-t border-neutral-200/40 dark:border-neutral-700/40"
                >
                  <span class="text-neutral-400">Default:</span>
                  <span class="font-mono text-neutral-600 dark:text-neutral-300"
                    >Disabled (WASM SIMD128)</span
                  >
                </div>
              </div>

              <!-- Browser Compatibility Status Card -->
              <div
                class="p-2.5 rounded-lg border text-[11px]"
                :class="
                  isWebGpuSupported
                    ? 'border-emerald-200 dark:border-emerald-900/60 bg-emerald-50/50 dark:bg-emerald-950/20'
                    : 'border-amber-200 dark:border-amber-900/60 bg-amber-50/50 dark:bg-amber-950/20'
                "
              >
                <div class="flex items-center gap-2 mb-1">
                  <span
                    class="w-2 h-2 rounded-full shrink-0"
                    :class="isWebGpuSupported ? 'bg-emerald-500' : 'bg-amber-500'"
                  />
                  <span
                    class="font-semibold"
                    :class="
                      isWebGpuSupported
                        ? 'text-emerald-800 dark:text-emerald-300'
                        : 'text-amber-800 dark:text-amber-300'
                    "
                  >
                    {{
                      isWebGpuSupported
                        ? 'WebGPU Supported in Browser'
                        : 'WebGPU Unavailable in Browser'
                    }}
                  </span>
                </div>
                <p
                  class="text-[10px]"
                  :class="
                    isWebGpuSupported
                      ? 'text-emerald-700 dark:text-emerald-400'
                      : 'text-amber-700 dark:text-amber-400'
                  "
                >
                  {{
                    isWebGpuSupported
                      ? 'navigator.gpu adapter is available. Toggling WebGPU will execute compute shaders directly on your GPU hardware.'
                      : 'Your current browser or device environment does not expose navigator.gpu. Inference will automatically execute using vectorized CPU WebAssembly SIMD128.'
                  }}
                </p>
              </div>

              <!-- Architecture Details -->
              <div class="space-y-1.5 pt-1 border-t border-neutral-100 dark:border-neutral-800">
                <span class="font-semibold text-[11px] text-neutral-700 dark:text-neutral-300">
                  Runtime Architecture
                </span>
                <div class="text-[10px] space-y-1.5 text-neutral-500 dark:text-neutral-400">
                  <div class="flex items-start gap-1.5">
                    <span class="font-semibold text-neutral-700 dark:text-neutral-300 shrink-0"
                      >WASM SIMD128:</span
                    >
                    <span
                      >100% pure Rust compiled to WebAssembly. Zero dependencies, portable across
                      all modern browsers and Node.js.</span
                    >
                  </div>
                  <div class="flex items-start gap-1.5">
                    <span class="font-semibold text-neutral-700 dark:text-neutral-300 shrink-0"
                      >WebGPU Backend:</span
                    >
                    <span
                      >Pluggable WGPU/WGSL compute shader pipeline. Offloads batch GEMM operations
                      to GPU hardware for reduced latency.</span
                    >
                  </div>
                </div>
              </div>
            </template>

            <!-- Tab 7: Model Weights & Hugging Face Hub -->
            <template v-else-if="configTab === 'models'">
              <div class="space-y-3">
                <!-- Selected Model Details Card -->
                <div
                  class="p-2.5 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 space-y-1.5"
                >
                  <div class="flex items-center justify-between">
                    <span class="font-semibold text-neutral-900 dark:text-neutral-100 text-xs">
                      {{ currentModel.name }}
                    </span>
                    <span
                      class="text-[10px] font-mono px-2 py-0.5 rounded bg-primary/10 text-primary font-semibold"
                    >
                      {{ currentModel.size }}
                    </span>
                  </div>
                  <p class="text-[11px] text-neutral-500 dark:text-neutral-400">
                    {{ currentModel.description }}
                  </p>
                  <div class="flex items-center gap-2 pt-1 text-[10px]">
                    <span class="text-neutral-400">Architecture:</span>
                    <code
                      class="px-1.5 py-0.5 rounded bg-neutral-200 dark:bg-neutral-800 font-mono"
                      >{{ currentModel.preset }}</code
                    >
                    <span class="text-neutral-400">Rec Batch:</span>
                    <code
                      class="px-1.5 py-0.5 rounded bg-neutral-200 dark:bg-neutral-800 font-mono"
                      >{{ currentModel.recommendedBatchSize }}</code
                    >
                  </div>
                </div>

                <!-- Hugging Face Hub Info & Links -->
                <div
                  class="p-2.5 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 space-y-2"
                >
                  <div class="flex items-center justify-between">
                    <span class="text-[11px] font-semibold text-neutral-700 dark:text-neutral-300">
                      Hugging Face Hub Repository
                    </span>
                    <a
                      :href="`https://huggingface.co/${HUGGINGFACE_REPO}`"
                      target="_blank"
                      rel="noopener noreferrer"
                      class="text-[10px] text-primary hover:underline font-mono inline-flex items-center gap-1"
                    >
                      <span>{{ HUGGINGFACE_REPO }}</span>
                      <svg class="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                        <path
                          fill-rule="evenodd"
                          d="M4.25 5.5a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 0 0 .75-.75v-4a.75.75 0 0 1 1.5 0v4A2.25 2.25 0 0 1 12.75 17h-8.5A2.25 2.25 0 0 1 2 14.75v-8.5A2.25 2.25 0 0 1 4.25 4h4a.75.75 0 0 1 0 1.5h-4Zm6.75-2a.75.75 0 0 1 .75-.75h5.5c.414 0 .75.336.75.75v5.5a.75.75 0 0 1-1.5 0V4.56l-6.72 6.72a.75.75 0 1 1-1.06-1.06l6.72-6.72h-4.44a.75.75 0 0 1-.75-.75Z"
                          clip-rule="evenodd"
                        />
                      </svg>
                    </a>
                  </div>
                  <div
                    class="text-[10px] font-mono space-y-1.5 text-neutral-600 dark:text-neutral-400"
                  >
                    <div>
                      <span
                        class="text-neutral-400 block text-[9px] uppercase tracking-wider font-semibold"
                        >Detection Weight URL</span
                      >
                      <span class="break-all select-all text-neutral-800 dark:text-neutral-200">{{
                        getModelCandidateUrls(selectedModel, baseURL).detection[0]
                      }}</span>
                    </div>
                    <div>
                      <span
                        class="text-neutral-400 block text-[9px] uppercase tracking-wider font-semibold"
                        >Recognition Weight URL</span
                      >
                      <span class="break-all select-all text-neutral-800 dark:text-neutral-200">{{
                        getModelCandidateUrls(selectedModel, baseURL).recognition[0]
                      }}</span>
                    </div>
                    <div>
                      <span
                        class="text-neutral-400 block text-[9px] uppercase tracking-wider font-semibold"
                        >Vocabulary / Dictionary URL</span
                      >
                      <span class="break-all select-all text-neutral-800 dark:text-neutral-200">{{
                        getModelCandidateUrls(selectedModel, baseURL).dictionary[0]
                      }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </div>
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
    <div
      class="grid grid-cols-1 lg:grid-cols-12 min-h-96 lg:h-[680px] xl:h-[740px] overflow-hidden"
    >
      <!-- Left: Interactive Image Canvas & Overlays (7 cols) -->
      <div
        class="lg:col-span-7 p-3.5 sm:p-4 flex flex-col items-center justify-center bg-neutral-950/5 dark:bg-black/30 border-b lg:border-b-0 lg:border-r border-neutral-200 dark:border-neutral-800 h-full max-h-[500px] lg:max-h-full overflow-auto"
      >
        <div
          class="relative max-w-full inline-block rounded-lg shadow-md overflow-hidden bg-white dark:bg-neutral-950"
        >
          <img
            ref="imgRef"
            :src="imageSrc"
            alt="OCR Target Document"
            @load="onImageLoad"
            class="block max-w-full max-h-[440px] lg:max-h-[580px] w-auto h-auto object-contain"
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

        <p class="mt-2 text-[11px] text-neutral-400">
          Tip: Hover over bounding boxes to highlight recognized text. Click to copy.
        </p>
      </div>

      <!-- Right: Inspector & Output Panel (5 cols) -->
      <div
        class="lg:col-span-5 flex flex-col bg-white dark:bg-neutral-900 h-full min-h-0 overflow-hidden"
      >
        <!-- Granularity & Tab Header -->
        <div
          class="p-3 border-b border-neutral-200 dark:border-neutral-800 flex items-center justify-between gap-2"
        >
          <div class="flex items-center gap-1">
            <button
              v-for="mode in ['lines', 'words'] as const"
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
                  : activeTab === 'tsv'
                    ? tsvExport
                    : activeTab === 'json'
                      ? jsonExport
                      : activeTab === 'config'
                        ? activeConfigJson
                        : results.map((r) => r.text).join('\n')
              )
            "
            class="px-2.5 py-1 text-[11px] rounded border border-neutral-200 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-all cursor-pointer"
          >
            {{ copied ? '✓ Copied' : 'Copy All' }}
          </button>
        </div>

        <!-- Secondary Tabs: Structured List / Spatial / CSV / JSON / Active Config -->
        <div
          class="flex border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50 text-[11px] overflow-x-auto"
        >
          <button
            type="button"
            @click="activeTab = 'structured'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer whitespace-nowrap"
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
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer whitespace-nowrap"
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
            @click="activeTab = 'tsv'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5"
            :class="
              activeTab === 'tsv'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            <span>TSV</span>
            <span class="text-[9px] px-1 py-0.2 rounded bg-neutral-200/80 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 font-mono">Sheets/Excel</span>
          </button>
          <button
            type="button"
            @click="activeTab = 'json'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer whitespace-nowrap"
            :class="
              activeTab === 'json'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            JSON
          </button>
          <button
            type="button"
            @click="activeTab = 'config'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5"
            :class="
              activeTab === 'config'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            <span>Config</span>
            <span
              v-if="isConfigModifiedSinceRun"
              class="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0"
              title="Settings modified since last run"
            />
          </button>
        </div>

        <!-- Tab Body Content -->
        <div
          class="flex-1 min-h-0 p-3 overflow-y-auto max-h-[460px] lg:max-h-full text-xs font-mono"
        >
          <!-- OCR Error Message Box -->
          <div
            v-if="ocrError"
            class="mb-3 p-3.5 rounded-xl border border-red-200 dark:border-red-900/80 bg-red-50 dark:bg-red-950/60 text-red-700 dark:text-red-300 flex items-start gap-3 shadow-xs select-text"
            role="alert"
          >
            <div
              class="p-1 rounded-md bg-red-100 dark:bg-red-900/60 text-red-600 dark:text-red-400 shrink-0 mt-0.5"
            >
              <svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                <path
                  fill-rule="evenodd"
                  d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM8.28 7.22a.75.75 0 0 0-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 1 0 1.06 1.06L10 11.06l1.72 1.72a.75.75 0 1 0 1.06-1.06L11.06 10l1.72-1.72a.75.75 0 0 0-1.06-1.06L10 8.94 8.28 7.22Z"
                  clip-rule="evenodd"
                />
              </svg>
            </div>
            <div class="flex-1 min-w-0">
              <h4 class="text-xs font-semibold text-red-800 dark:text-red-200">
                OCR Execution Error
              </h4>
              <pre
                class="mt-1 text-[11px] font-mono whitespace-pre-wrap break-all text-red-600 dark:text-red-400 select-text"
                >{{ ocrError }}</pre>
            </div>
            <button
              type="button"
              @click="ocrError = null"
              class="text-red-400 hover:text-red-600 dark:hover:text-red-200 p-0.5 rounded cursor-pointer"
              title="Dismiss error"
            >
              <svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                <path
                  d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z"
                />
              </svg>
            </button>
          </div>

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

            <!-- Active Config skeleton -->
            <div
              v-else-if="activeTab === 'config'"
              class="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 space-y-3 animate-pulse"
            >
              <div class="h-4 rounded bg-neutral-200 dark:bg-neutral-800 w-1/3" />
              <div class="h-16 rounded bg-neutral-200 dark:bg-neutral-800 w-full" />
              <div class="h-24 rounded bg-neutral-200 dark:bg-neutral-800 w-full" />
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

            <!-- TSV view with Table Grid & Raw toggle -->
            <div v-else-if="activeTab === 'tsv'" class="space-y-2">
              <div class="flex items-center justify-between pb-1 text-[11px]">
                <div class="flex items-center gap-1 bg-neutral-100 dark:bg-neutral-900 p-0.5 rounded-md border border-neutral-200 dark:border-neutral-800">
                  <button
                    type="button"
                    @click="tsvViewMode = 'table'"
                    class="px-2 py-0.5 rounded transition-all cursor-pointer"
                    :class="tsvViewMode === 'table' ? 'bg-white dark:bg-neutral-800 font-medium text-neutral-900 dark:text-neutral-100 shadow-sm' : 'text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'"
                  >
                    Table Grid
                  </button>
                  <button
                    type="button"
                    @click="tsvViewMode = 'raw'"
                    class="px-2 py-0.5 rounded transition-all cursor-pointer"
                    :class="tsvViewMode === 'raw' ? 'bg-white dark:bg-neutral-800 font-medium text-neutral-900 dark:text-neutral-100 shadow-sm' : 'text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'"
                  >
                    Raw TSV
                  </button>
                </div>
                <span class="text-[10px] text-neutral-500 dark:text-neutral-400">
                  Tab-delimited: ready to paste into Sheets / Excel
                </span>
              </div>

              <!-- Table Grid View -->
              <div v-if="tsvViewMode === 'table'" class="overflow-x-auto border border-neutral-200 dark:border-neutral-800 rounded-lg shadow-sm">
                <table class="w-full text-left text-[11px] border-collapse">
                  <thead>
                    <tr class="bg-neutral-100/80 dark:bg-neutral-900/80 border-b border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400 font-semibold uppercase tracking-wider text-[10px]">
                      <th class="py-1.5 px-2.5 w-10">#</th>
                      <th class="py-1.5 px-2.5">Text</th>
                      <th class="py-1.5 px-2.5 w-20">Score</th>
                      <th class="py-1.5 px-2.5 w-36">Box (X, Y, W, H)</th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-neutral-100 dark:divide-neutral-800/60 font-sans">
                    <tr
                      v-for="(row, idx) in results"
                      :key="idx"
                      @mouseenter="hoveredIndex = idx"
                      @mouseleave="hoveredIndex = null"
                      class="transition-colors cursor-pointer hover:bg-primary/5 dark:hover:bg-primary/10"
                      :class="hoveredIndex === idx ? 'bg-primary/10 dark:bg-primary/20' : ''"
                    >
                      <td class="py-1.5 px-2.5 text-neutral-400 font-mono text-[10px]">{{ idx + 1 }}</td>
                      <td class="py-1.5 px-2.5 font-medium text-neutral-800 dark:text-neutral-200 select-text">{{ row.text }}</td>
                      <td class="py-1.5 px-2.5 font-mono text-[10px]">
                        <span
                          class="px-1.5 py-0.5 rounded text-[10px] font-medium"
                          :class="row.score >= 0.9 ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-400' : row.score >= 0.75 ? 'bg-blue-100 text-blue-800 dark:bg-blue-950/60 dark:text-blue-400' : 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-400'"
                        >
                          {{ (row.score * 100).toFixed(1) }}%
                        </span>
                      </td>
                      <td class="py-1.5 px-2.5 text-neutral-500 dark:text-neutral-400 font-mono text-[10px]">
                        {{ Math.round(row.frame.left) }}, {{ Math.round(row.frame.top) }}, {{ Math.round(row.frame.width) }}, {{ Math.round(row.frame.height) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <!-- Raw TSV view -->
              <pre
                v-else
                class="whitespace-pre overflow-x-auto text-neutral-700 dark:text-neutral-300 p-3 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 font-mono text-xs leading-relaxed select-text"
                style="
                  font-family:
                    ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono',
                    'Courier New', monospace;
                  tab-size: 4;
                "
              >{{ tsvExport }}</pre>
            </div>

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

            <!-- Active Run Configuration view -->
            <OcrActiveConfigView
              v-else-if="activeTab === 'config'"
              :active-run="activeRunConfig"
              :current-ocr-config="ocrConfig"
              :current-model-name="currentModel.name"
              :current-output-mode="outputMode"
              :is-modified-since-run="isConfigModifiedSinceRun"
            />
          </template>
        </div>

        <!-- Active Config Summary Footer -->
        <div
          class="mt-auto px-3 py-1.5 border-t border-neutral-200 dark:border-neutral-800 bg-neutral-50/70 dark:bg-neutral-950/70 flex items-center justify-between text-[11px] shrink-0"
        >
          <div class="flex items-center gap-1.5 min-w-0 mr-2">
            <span
              v-if="activeRunConfig"
              class="truncate font-mono text-[10px] text-neutral-700 dark:text-neutral-300"
              :title="`${activeRunConfig.modelName} • ${activeRunConfig.outputMode === 'words' ? 'asWords()' : 'asLines()'} • score ≥ ${activeScorePercent}%`"
            >
              {{ activeRunConfig.modelName }} •
              {{ activeRunConfig.outputMode === 'words' ? 'asWords()' : 'asLines()' }} • score ≥
              {{ activeScorePercent }}%
            </span>
            <span v-else class="text-neutral-400 italic">No run yet</span>
            <span
              v-if="isConfigModifiedSinceRun"
              class="inline-flex items-center text-[9px] px-1.5 py-0.2 rounded bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-400 shrink-0 font-medium"
              title="Configuration modified in settings - click Run OCR to apply"
            >
              modified
            </span>
          </div>
          <button
            type="button"
            @click="activeTab = 'config'"
            class="text-primary hover:underline font-medium shrink-0 cursor-pointer text-[11px]"
          >
            {{ activeTab === 'config' ? 'Viewing' : 'Details' }}
          </button>
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
