<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useRuntimeConfig } from '#app';
import { useOcrDemoI18n } from '../../composables/useOcrDemoI18n';
import { useWebOcrExecution } from '../../composables/useWebOcrExecution';
import { useWebOcrSettings } from '../../composables/useWebOcrSettings';
import { availableModels, defaultModel, type OcrModelOption } from '../../types/ocrModels';
import { getSampleImages, type SampleImage } from '../../types/ocrSamples';
import { fetchHuggingFaceModelIndex } from '../../utils/ocrReleaseModels';
import { isRunConfigModified } from '../../utils/ocrResultFormatter';
import OcrLoadingOverlay from '../OcrLoadingOverlay.vue';
import OcrDemoToolbar from '../ocr/OcrDemoToolbar.vue';
import OcrImageCanvas from '../ocr/OcrImageCanvas.vue';
import OcrResultsPanel from '../ocr/OcrResultsPanel.vue';
import OcrSampleBar from '../ocr/OcrSampleBar.vue';
import OcrSettingsModal from '../ocr/OcrSettingsModal.vue';

const rawBaseURL = useRuntimeConfig().app.baseURL || '/';
const baseURL = rawBaseURL.endsWith('/') ? rawBaseURL : `${rawBaseURL}/`;

const { strings } = useOcrDemoI18n();
const samples = getSampleImages(baseURL);
const defaultSample = samples[0]!;

const {
  ocrConfig,
  isConfigOpen,
  configTab,
  isCustomConfig,
  toggleConfigPopover,
  openConfigTab,
  resetConfigDefaults,
  applyPreset,
  isPresetActive,
} = useWebOcrSettings();

const selectedModel = ref<string>('ppocrv6-tiny');
const currentModel = computed<OcrModelOption>(
  () => availableModels.find((m) => m.id === selectedModel.value) ?? defaultModel
);

const selectedSample = ref<string>(defaultSample.id);
const customFileName = ref<string | null>(null);
const imageSrc = ref<string>(defaultSample.url);
const outputMode = ref<'lines' | 'words'>('lines');
const isWebGpuSupported = ref<boolean>(false);
const isExpanded = ref<boolean>(false);

let customObjectUrl: string | null = null;

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

const isConfigModifiedSinceRun = computed<boolean>(() =>
  isRunConfigModified(activeRunConfig.value, selectedModel.value, outputMode.value, ocrConfig.value)
);

function revokeCustomUrl() {
  if (customObjectUrl) {
    URL.revokeObjectURL(customObjectUrl);
    customObjectUrl = null;
  }
}

function pickSample(sample: SampleImage) {
  resetResults();
  revokeCustomUrl();
  customFileName.value = null;
  selectedSample.value = sample.id;
  imageSrc.value = sample.url;
}

function loadCustomImageFile(file: File) {
  resetResults();
  revokeCustomUrl();
  const name = file.name || 'pasted-image.png';
  customObjectUrl = URL.createObjectURL(file);
  customFileName.value = name;
  selectedSample.value = 'custom';
  imageSrc.value = `${customObjectUrl}#${encodeURIComponent(name)}`;
}

function setGranularity(mode: 'lines' | 'words') {
  if (outputMode.value === mode) return;
  outputMode.value = mode;
  if (activeRunConfig.value) {
    runOcr();
  }
}

async function copySingleItem(text: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    console.error('Copy item failed:', err);
  }
}

function handleApplyAndRun() {
  isConfigOpen.value = false;
  runOcr();
}

function onPaste(event: ClipboardEvent) {
  const target = event.target as HTMLElement | null;
  if (
    target &&
    (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
  ) {
    return;
  }
  const items = event.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) {
        event.preventDefault();
        loadCustomImageFile(file);
        break;
      }
    }
  }
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    if (isConfigOpen.value) {
      isConfigOpen.value = false;
    } else if (isExpanded.value) {
      isExpanded.value = false;
    }
    return;
  }
  if (
    (event.ctrlKey || event.metaKey) &&
    event.key === 'Enter' &&
    !isScanning.value &&
    status.value !== 'initializing'
  ) {
    event.preventDefault();
    runOcr();
  }
}

watch(isExpanded, (expanded) => {
  if (typeof document === 'undefined') return;
  document.body.style.overflow = expanded ? 'hidden' : '';
});

onMounted(() => {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu)) {
    isWebGpuSupported.value = true;
  }
  status.value = 'idle';
  statusMessage.value = `${currentModel.value.name} (${currentModel.value.size})`;
  fetchHuggingFaceModelIndex().catch(() => {});

  if (typeof window !== 'undefined') {
    window.addEventListener('keydown', onKeydown);
    window.addEventListener('paste', onPaste);
  }
});

onUnmounted(() => {
  revokeCustomUrl();
  if (typeof document !== 'undefined') {
    document.body.style.overflow = '';
  }
  if (typeof window !== 'undefined') {
    window.removeEventListener('keydown', onKeydown);
    window.removeEventListener('paste', onPaste);
  }
});
</script>

<template>
  <div>
    <div
      v-if="isExpanded"
      class="fixed inset-0 z-40 bg-black/50 backdrop-blur-xs"
      @click="isExpanded = false"
    />

    <div
      class="rusto-demo-container not-prose border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 shadow-lg overflow-hidden flex flex-col"
      :class="
        isExpanded
          ? 'fixed inset-2 sm:inset-5 z-50 rounded-2xl shadow-2xl'
          : 'relative my-5 rounded-2xl'
      "
    >
      <OcrLoadingOverlay :progress="progress" />

      <OcrDemoToolbar
        :selected-model="selectedModel"
        :current-model="currentModel"
        :status="status"
        :status-message="statusMessage"
        :is-scanning="isScanning"
        :duration-ms="durationMs"
        :backend="ocrConfig.inferenceBackend"
        :use-web-gpu="ocrConfig.useWebGpu"
        :is-config-open="isConfigOpen"
        :is-custom-config="isCustomConfig"
        :is-config-modified="isConfigModifiedSinceRun"
        :is-expanded="isExpanded"
        :strings="strings.toolbar"
        @update:selected-model="switchModel"
        @retry-init="initEngine(selectedModel)"
        @open-runtime-settings="openConfigTab('runtime')"
        @toggle-settings="toggleConfigPopover"
        @toggle-expand="isExpanded = !isExpanded"
        @run-ocr="runOcr"
      />

      <OcrSampleBar
        :samples="samples"
        :selected-sample="selectedSample"
        :custom-file-name="customFileName"
        :disabled="status === 'initializing' || isScanning"
        :strings="strings.samples"
        @select-sample="pickSample"
        @upload-file="loadCustomImageFile"
      />

      <div
        class="grid grid-cols-1 lg:grid-cols-12 overflow-hidden"
        :class="isExpanded ? 'flex-1 min-h-0' : 'min-h-96 lg:h-[620px] xl:h-[660px]'"
      >
        <OcrImageCanvas
          v-model:hovered-index="hoveredIndex"
          :image-src="imageSrc"
          :is-scanning="isScanning"
          :results="results"
          :has-run="activeRunConfig !== null"
          :strings="strings.canvas"
          @copy-item="copySingleItem"
          @upload-file="loadCustomImageFile"
        />

        <OcrResultsPanel
          v-model:hovered-index="hoveredIndex"
          :results="results"
          :spatial-text="spatialText"
          :output-mode="outputMode"
          :is-scanning="isScanning"
          :ocr-error="ocrError"
          :active-run-config="activeRunConfig"
          :ocr-config="ocrConfig"
          :current-model="currentModel"
          :is-config-modified="isConfigModifiedSinceRun"
          :strings="strings.results"
          @change-granularity="setGranularity"
          @dismiss-error="ocrError = null"
          @open-settings="openConfigTab('detection')"
          @run-ocr="runOcr"
        />
      </div>

      <OcrSettingsModal
        v-model:active-tab="configTab"
        :is-open="isConfigOpen"
        :config="ocrConfig"
        :current-model="currentModel"
        :selected-model-id="selectedModel"
        :base-u-r-l="baseURL"
        :is-web-gpu-supported="isWebGpuSupported"
        :is-config-modified="isConfigModifiedSinceRun"
        :has-run="activeRunConfig !== null"
        :is-preset-active="isPresetActive"
        :strings="strings.settings"
        @close="isConfigOpen = false"
        @reset-defaults="resetConfigDefaults"
        @apply-preset="applyPreset"
        @layout-model-changed="resetResults"
        @backend-changed="resetResults"
        @apply-and-run="handleApplyAndRun"
      />
    </div>
  </div>
</template>
