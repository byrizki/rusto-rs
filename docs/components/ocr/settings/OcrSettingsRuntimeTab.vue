<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { OcrConfig } from '../../../types/ocrConfig';
import type { OcrModelOption } from '../../../types/ocrModels';
import {
  getModelCandidateUrls,
  HUGGINGFACE_REPO,
  isModelSupportedByBackend,
} from '../../../utils/ocrReleaseModels';
import OcrInfoPopover from './OcrInfoPopover.vue';
import OcrModelSelect from './OcrModelSelect.vue';
import OcrSettingsSubTabs, { type SettingsSubTab } from './OcrSettingsSubTabs.vue';

type RuntimeSection = 'engines' | 'model' | 'weights';

const props = defineProps<{
  config: OcrConfig;
  currentModel: OcrModelOption;
  selectedModelId: string;
  baseURL: string;
  isWebGpuSupported: boolean;
  subTab?: RuntimeSection;
}>();

const emit = defineEmits<{
  selectModel: [modelId: string];
  backendChanged: [backend: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu'];
  'update:subTab': [tab: RuntimeSection];
}>();

const activeSection = ref<RuntimeSection>(props.subTab ?? 'engines');

watch(
  () => props.subTab,
  (val) => {
    if (val) activeSection.value = val;
  }
);

watch(activeSection, (val) => {
  emit('update:subTab', val);
});

watch(
  () => props.selectedModelId,
  (modelId) => {
    if (!isModelSupportedByBackend(modelId, props.config.inferenceBackend)) {
      props.config.inferenceBackend = 'rusto-nn';
      props.config.useWebGpu = false;
      emit('backendChanged', 'rusto-nn');
    }
  }
);
const copiedKey = ref<string | null>(null);

const subTabs = computed<SettingsSubTab[]>(() => [
  {
    id: 'engines',
    label: 'Engine Backend',
    badge: props.config.inferenceBackend,
    description: 'Pluggable inference runtimes compiled to separate WebAssembly binaries',
  },
  {
    id: 'model',
    label: 'Model Info',
    badge: props.currentModel.id,
    description: 'Active model architecture, precision, and runtime characteristics',
  },
  {
    id: 'weights',
    label: 'Weights & Assets',
    badge: 'Hugging Face',
    description: 'Remote CDN and local asset fallback URLs for weights & dictionaries',
  },
]);

const candidateUrls = computed(() =>
  getModelCandidateUrls(props.selectedModelId, props.baseURL, props.config.inferenceBackend)
);

function isEngineCompatible(id: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu'): boolean {
  return isModelSupportedByBackend(props.selectedModelId, id);
}

interface EngineOption {
  id: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu';
  name: string;
  label: string;
  description: string;
  wasmFile: string;
  requiresWebGpu?: boolean;
}

const engines: EngineOption[] = [
  {
    id: 'rusto-nn',
    name: 'rusto-nn',
    label: 'Default Pure-Rust MNN',
    description:
      '100% pure-Rust Alibaba MNN execution graph. Zero native C++ bloat, ultra-low memory, optimal SIMD.',
    wasmFile: 'rusto_nn_bg.wasm',
  },
  {
    id: 'rusto-rten',
    name: 'rusto-rten',
    label: 'Pure-Rust RTen',
    description:
      'Pure-Rust neural network inference engine optimized for FP32 ONNX & RTen tensor models (FP16 and INT8 are not supported).',
    wasmFile: 'rusto_rten_bg.wasm',
  },
  {
    id: 'rusto-webgpu',
    name: 'rusto-webgpu',
    label: 'WebGPU Hardware Compute',
    description:
      'Offloads matrix & convolution ops to GPU via WGSL compute shaders (rusto-nn-webgpu).',
    wasmFile: 'rusto_webgpu_bg.wasm',
    requiresWebGpu: true,
  },
];

function selectEngine(id: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu') {
  if (!isEngineCompatible(id)) return;
  props.config.inferenceBackend = id;
  props.config.useWebGpu = id === 'rusto-webgpu';
  emit('backendChanged', id);
}

async function copyUrl(key: string, url?: string) {
  if (!url) return;
  try {
    await navigator.clipboard.writeText(url);
    copiedKey.value = key;
    setTimeout(() => {
      if (copiedKey.value === key) copiedKey.value = null;
    }, 1500);
  } catch {
    // Clipboard unavailable in restricted contexts
  }
}
</script>

<template>
  <div class="space-y-3">
    <div
      data-setting-id="modelSelection"
      class="pb-3 border-b border-zinc-200/70 dark:border-zinc-800"
    >
      <OcrModelSelect
        :selected-model-id="selectedModelId"
        :current-model="currentModel"
        :inference-backend="config.inferenceBackend"
        @select="(modelId) => emit('selectModel', modelId)"
      />
    </div>

    <OcrSettingsSubTabs v-model="activeSection" :tabs="subTabs" class="text-xs">
      <!-- Pluggable Inference Engines -->
      <template v-if="activeSection === 'engines'">
        <div data-setting-id="inferenceBackend" class="space-y-2">
          <div
            v-for="engine in engines"
            :key="engine.id"
            @click="selectEngine(engine.id)"
            class="p-2.5 rounded-lg border transition-all relative"
            :class="[
              isEngineCompatible(engine.id)
                ? config.inferenceBackend === engine.id
                  ? 'cursor-pointer border-primary bg-primary/5 dark:bg-primary/10'
                  : 'cursor-pointer border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 hover:border-zinc-300 dark:hover:border-zinc-700'
                : 'opacity-55 cursor-not-allowed border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50',
            ]"
          >
            <div class="flex items-center justify-between gap-2">
              <div class="flex items-center gap-1.5">
                <input
                  type="radio"
                  :name="'ocr-inference-engine'"
                  :value="engine.id"
                  :checked="config.inferenceBackend === engine.id"
                  :disabled="!isEngineCompatible(engine.id)"
                  class="accent-primary cursor-pointer disabled:cursor-not-allowed w-3.5 h-3.5 shrink-0"
                  @change="selectEngine(engine.id)"
                />
                <span class="font-mono font-semibold text-zinc-900 dark:text-zinc-100 text-xs">
                  {{ engine.name }}
                </span>
                <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
                  {{ engine.label }}
                </span>
                <OcrInfoPopover :text="engine.description" />
              </div>
              <span
                class="text-[10px] font-mono px-1.5 py-0.5 rounded border"
                :class="
                  isEngineCompatible(engine.id)
                    ? 'border-zinc-200 dark:border-zinc-800 text-zinc-500 dark:text-zinc-400'
                    : 'border-amber-300/70 dark:border-amber-800/70 bg-amber-50/70 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300'
                "
              >
                {{ isEngineCompatible(engine.id) ? engine.wasmFile : 'FP32 only (No FP16/INT8)' }}
              </span>
            </div>

            <p
              v-if="!isEngineCompatible(engine.id)"
              class="mt-1.5 pl-5 text-[10px] font-mono text-amber-600 dark:text-amber-400"
            >
              RTen does not support FP16 or INT8 models. Select an FP32 model to enable rusto-rten.
            </p>

            <div
              v-if="engine.requiresWebGpu"
              class="mt-1.5 pl-5 text-[10px] font-mono"
              :class="
                isWebGpuSupported
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-amber-600 dark:text-amber-400'
              "
            >
              {{
                isWebGpuSupported
                  ? '● WebGPU adapter detected'
                  : '○ WebGPU not detected (CPU SIMD128 fallback)'
              }}
            </div>
          </div>
        </div>
      </template>

      <!-- Active Model Info -->
      <template v-else-if="activeSection === 'model'">
        <div data-setting-id="modelInfo" class="space-y-2 text-[11px]">
          <div
            class="grid grid-cols-2 gap-x-4 gap-y-1.5 py-1 border-b border-zinc-100 dark:border-zinc-800/80"
          >
            <div>
              <span class="text-zinc-400 block text-[10px] uppercase">Model ID</span>
              <code class="font-mono font-semibold text-zinc-900 dark:text-zinc-100">{{
                currentModel.id
              }}</code>
            </div>
            <div>
              <span class="text-zinc-400 block text-[10px] uppercase">Preset / Family</span>
              <code class="font-mono font-semibold text-zinc-900 dark:text-zinc-100"
                >{{ currentModel.preset }} · {{ currentModel.categoryLabel }}</code
              >
            </div>
            <div>
              <span class="text-zinc-400 block text-[10px] uppercase">Precision / Trait</span>
              <span class="font-medium text-zinc-800 dark:text-zinc-200">{{
                currentModel.trait
              }}</span>
            </div>
            <div>
              <span class="text-zinc-400 block text-[10px] uppercase">Weight Size &amp; Batch</span>
              <span class="font-mono text-zinc-800 dark:text-zinc-200"
                >{{ currentModel.size }} · batch={{ currentModel.recommendedBatchSize }}</span
              >
            </div>
          </div>

          <p class="text-zinc-600 dark:text-zinc-400 leading-relaxed">
            {{ currentModel.description }}
          </p>

          <div v-if="currentModel.tags?.length" class="flex items-center gap-1.5 flex-wrap pt-1">
            <span
              v-for="tag in currentModel.tags"
              :key="tag"
              class="px-1.5 py-0.5 text-[10px] font-mono rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300"
            >
              {{ tag }}
            </span>
          </div>
        </div>
      </template>

      <!-- Weights Sources -->
      <template v-else>
        <div data-setting-id="weightsAssets" class="space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">
              Repository
            </span>
            <a
              :href="`https://huggingface.co/${HUGGINGFACE_REPO}`"
              target="_blank"
              rel="noopener noreferrer"
              class="text-[11px] text-primary hover:underline font-mono inline-flex items-center gap-1"
            >
              <span>{{ HUGGINGFACE_REPO }}</span>
              <span aria-hidden="true">↗</span>
            </a>
          </div>
          <div class="space-y-1.5 text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
            <div
              v-for="item in [
                { key: 'det', label: 'det', url: candidateUrls.detection[0] },
                { key: 'rec', label: 'rec', url: candidateUrls.recognition[0] },
                { key: 'dict', label: 'dict', url: candidateUrls.dictionary[0] },
              ]"
              :key="item.key"
              class="flex items-center justify-between gap-2 bg-zinc-50 dark:bg-zinc-950/60 px-2 py-1 rounded border border-zinc-200/60 dark:border-zinc-800"
            >
              <div class="truncate" :title="item.url">
                <span class="text-zinc-400">{{ item.label }}:</span> {{ item.url }}
              </div>
              <button
                type="button"
                @click="copyUrl(item.key, item.url)"
                class="shrink-0 text-[10px] text-zinc-400 hover:text-primary cursor-pointer"
                :title="`Copy ${item.label} URL`"
              >
                {{ copiedKey === item.key ? 'Copied' : 'Copy' }}
              </button>
            </div>
          </div>
        </div>
      </template>
    </OcrSettingsSubTabs>
  </div>
</template>
