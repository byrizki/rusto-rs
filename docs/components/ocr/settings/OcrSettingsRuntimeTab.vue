<script setup lang="ts">
import { computed } from 'vue';
import type { OcrConfig } from '../../../types/ocrConfig';
import type { OcrModelOption } from '../../../types/ocrModels';
import { getModelCandidateUrls, HUGGINGFACE_REPO } from '../../../utils/ocrReleaseModels';

const props = defineProps<{
  config: OcrConfig;
  currentModel: OcrModelOption;
  selectedModelId: string;
  baseURL: string;
  isWebGpuSupported: boolean;
}>();

const emit = defineEmits<{
  backendChanged: [backend: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu'];
}>();

const candidateUrls = computed(() => getModelCandidateUrls(props.selectedModelId, props.baseURL));

interface EngineOption {
  id: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu';
  name: string;
  badge: string;
  badgeClass: string;
  description: string;
  wasmFile: string;
  requiresWebGpu?: boolean;
}

const engines: EngineOption[] = [
  {
    id: 'rusto-nn',
    name: 'rusto-nn',
    badge: 'Default · Pure-Rust',
    badgeClass:
      'bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
    description:
      '100% pure-Rust Alibaba MNN execution graph. Zero native C++ bloat, ultra-low memory, optimal SIMD.',
    wasmFile: 'rusto_nn_bg.wasm',
  },
  {
    id: 'rusto-rten',
    name: 'rusto-rten',
    badge: 'RTen Engine',
    badgeClass:
      'bg-blue-50 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800',
    description:
      'Pure-Rust neural network inference engine optimized for ONNX & RTen tensor models.',
    wasmFile: 'rusto_rten_bg.wasm',
  },
  {
    id: 'rusto-webgpu',
    name: 'rusto-webgpu',
    badge: 'Hardware Compute',
    badgeClass:
      'bg-purple-50 dark:bg-purple-950/60 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-800',
    description:
      'Offloads matrix & convolution ops to GPU via WGSL compute shaders (rusto-nn-webgpu).',
    wasmFile: 'rusto_webgpu_bg.wasm',
    requiresWebGpu: true,
  },
];

function selectEngine(id: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu') {
  props.config.inferenceBackend = id;
  props.config.useWebGpu = id === 'rusto-webgpu';
  emit('backendChanged', id);
}
</script>

<template>
  <div class="space-y-3.5 text-xs">
    <!-- Pluggable Inference Engines -->
    <div
      class="p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50 space-y-2.5"
    >
      <div class="flex items-center justify-between">
        <div>
          <span class="font-semibold text-zinc-900 dark:text-zinc-100 block">
            Inference Engine Backend
          </span>
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
            Pluggable inference runtimes compiled to separate WebAssembly binaries
          </span>
        </div>
      </div>

      <div class="space-y-2">
        <div
          v-for="engine in engines"
          :key="engine.id"
          @click="selectEngine(engine.id)"
          class="p-2.5 rounded-lg border transition-all cursor-pointer relative"
          :class="[
            config.inferenceBackend === engine.id
              ? 'border-primary/80 bg-primary/5 dark:bg-primary/10 shadow-2xs ring-1 ring-primary/40'
              : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 hover:border-zinc-300 dark:hover:border-zinc-700',
          ]"
        >
          <div class="flex items-center justify-between gap-2 mb-1">
            <div class="flex items-center gap-2">
              <input
                type="radio"
                :name="'ocr-inference-engine'"
                :value="engine.id"
                :checked="config.inferenceBackend === engine.id"
                class="accent-primary cursor-pointer w-3.5 h-3.5 shrink-0"
                @change="selectEngine(engine.id)"
              />
              <span class="font-mono font-semibold text-zinc-900 dark:text-zinc-100 text-xs">
                {{ engine.name }}
              </span>
            </div>
            <span
              class="px-1.5 py-0.5 rounded text-[10px] font-medium border"
              :class="engine.badgeClass"
            >
              {{ engine.badge }}
            </span>
          </div>

          <p class="text-[11px] text-zinc-600 dark:text-zinc-400 leading-snug pl-5">
            {{ engine.description }}
          </p>

          <div
            class="flex items-center justify-between gap-2 mt-2 pt-1.5 border-t border-zinc-100 dark:border-zinc-800/80 text-[10px] font-mono text-zinc-400 pl-5"
          >
            <span>binary: {{ engine.wasmFile }}</span>
            <span
              v-if="engine.requiresWebGpu"
              class="text-[10px] font-sans"
              :class="
                isWebGpuSupported
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-amber-600 dark:text-amber-400'
              "
            >
              {{ isWebGpuSupported ? '● WebGPU supported' : '○ WebGPU not detected' }}
            </span>
          </div>
        </div>
      </div>

      <div
        v-if="config.inferenceBackend === 'rusto-webgpu' && !isWebGpuSupported"
        class="p-2 rounded-lg border border-amber-200/80 dark:border-amber-900/60 bg-amber-50/60 dark:bg-amber-950/30 text-amber-800 dark:text-amber-300 text-[11px] flex items-start gap-2"
      >
        <span class="w-2 h-2 rounded-full mt-1 shrink-0 bg-amber-500" />
        <span>
          WebGPU unavailable in this browser. When executing, the engine will automatically fall
          back to SIMD128 CPU compute.
        </span>
      </div>
    </div>

    <!-- Active Model Info -->
    <div
      class="p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50 space-y-2"
    >
      <div class="flex items-center justify-between gap-2">
        <span class="font-semibold text-zinc-900 dark:text-zinc-100">
          {{ currentModel.name }}
        </span>
        <span
          class="px-2 py-0.5 rounded-md text-[10px] font-mono bg-primary/10 text-primary font-semibold tabular-nums"
        >
          {{ currentModel.size }}
        </span>
      </div>
      <p class="text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug">
        {{ currentModel.description }}
      </p>
      <div class="flex flex-wrap items-center gap-2 pt-1 text-[11px]">
        <span class="px-2 py-0.5 rounded bg-zinc-200/70 dark:bg-zinc-800 font-mono text-[10px]">
          preset: {{ currentModel.preset }}
        </span>
        <span class="px-2 py-0.5 rounded bg-zinc-200/70 dark:bg-zinc-800 font-mono text-[10px]">
          {{ currentModel.trait }}
        </span>
      </div>
    </div>

    <!-- Weights Sources -->
    <div
      class="p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50 space-y-2"
    >
      <div class="flex items-center justify-between">
        <span class="text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">
          Model Weights Source
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
      <div class="space-y-1 text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
        <div class="truncate" :title="candidateUrls.detection[0]">
          <span class="text-zinc-400">det:</span> {{ candidateUrls.detection[0] }}
        </div>
        <div class="truncate" :title="candidateUrls.recognition[0]">
          <span class="text-zinc-400">rec:</span> {{ candidateUrls.recognition[0] }}
        </div>
        <div class="truncate" :title="candidateUrls.dictionary[0]">
          <span class="text-zinc-400">dict:</span> {{ candidateUrls.dictionary[0] }}
        </div>
      </div>
    </div>
  </div>
</template>
