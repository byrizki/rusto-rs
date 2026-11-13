<script setup lang="ts">
import { computed, ref } from 'vue';
import { buildDetectTextOptions, type ActiveRunConfig, type OcrConfig } from '../types/ocrConfig';
import { formatFluentBuilderSnippet, formatOptionsSnippet } from '../utils/ocrResultFormatter';

const props = defineProps<{
  activeRun: ActiveRunConfig | null;
  currentOcrConfig: OcrConfig;
  currentModelName: string;
  currentOutputMode: 'lines' | 'words';
  isModifiedSinceRun: boolean;
}>();

const copied = ref<boolean>(false);
const codeTab = ref<'builder' | 'options' | 'json'>('builder');

const displayConfig = computed(() => {
  const isSnapshot = Boolean(props.activeRun && !props.isModifiedSinceRun);
  const rawConfig =
    isSnapshot && props.activeRun ? props.activeRun.rawConfig : props.currentOcrConfig;
  const backendLabel =
    rawConfig.useWebGpu || rawConfig.inferenceBackend === 'rusto-webgpu'
      ? 'WebGPU'
      : rawConfig.inferenceBackend === 'rusto-rten'
        ? 'RTen'
        : 'WASM SIMD';

  if (isSnapshot && props.activeRun) {
    return {
      isSnapshot: true,
      modelName: props.activeRun.modelName,
      outputMode: props.activeRun.outputMode,
      durationMs: props.activeRun.durationMs,
      itemsCount: props.activeRun.itemsCount,
      backendLabel,
      options: props.activeRun.options,
    };
  }
  return {
    isSnapshot: Boolean(props.activeRun),
    modelName: props.currentModelName,
    outputMode: props.currentOutputMode,
    durationMs: props.activeRun?.durationMs ?? null,
    itemsCount: props.activeRun?.itemsCount ?? 0,
    backendLabel,
    options: buildDetectTextOptions(props.currentOcrConfig, props.currentOutputMode),
  };
});

const activeCodeSnippet = computed(() => {
  if (codeTab.value === 'builder') {
    return formatFluentBuilderSnippet(displayConfig.value.outputMode, displayConfig.value.options);
  }
  if (codeTab.value === 'options') {
    return formatOptionsSnippet(displayConfig.value.options);
  }
  return JSON.stringify(displayConfig.value.options, null, 2);
});

async function copyActiveCode() {
  try {
    await navigator.clipboard.writeText(activeCodeSnippet.value);
    copied.value = true;
    setTimeout(() => {
      copied.value = false;
    }, 1800);
  } catch (err) {
    console.error('Copy code failed:', err);
  }
}
</script>

<template>
  <div class="space-y-3 text-xs">
    <div
      class="px-3 py-2 rounded-xl border flex flex-wrap items-center justify-between gap-2"
      :class="
        isModifiedSinceRun
          ? 'border-amber-200 dark:border-amber-900/60 bg-amber-50/70 dark:bg-amber-950/30 text-amber-800 dark:text-amber-300'
          : displayConfig.isSnapshot
            ? 'border-emerald-200 dark:border-emerald-900/60 bg-emerald-50/70 dark:bg-emerald-950/30 text-emerald-800 dark:text-emerald-300'
            : 'border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 text-zinc-700 dark:text-zinc-300'
      "
    >
      <div class="flex items-center gap-2 font-medium">
        <span
          class="w-2 h-2 rounded-full shrink-0"
          :class="
            isModifiedSinceRun
              ? 'bg-amber-500'
              : displayConfig.isSnapshot
                ? 'bg-emerald-500'
                : 'bg-zinc-400'
          "
        />
        <span>
          {{
            isModifiedSinceRun
              ? 'Previewing updated settings (Re-run to apply)'
              : displayConfig.isSnapshot
                ? 'Synced with active scan'
                : 'Current pipeline configuration'
          }}
        </span>
      </div>

      <span
        v-if="displayConfig.durationMs !== null && !isModifiedSinceRun"
        class="font-mono text-[11px] tabular-nums"
      >
        {{ displayConfig.durationMs }} ms • {{ displayConfig.itemsCount }} items
      </span>
    </div>

    <div class="rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden">
      <div
        class="px-2.5 py-1.5 bg-zinc-100/80 dark:bg-zinc-950 border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-between gap-2"
      >
        <div class="flex items-center gap-1">
          <button
            type="button"
            @click="codeTab = 'builder'"
            class="px-2 py-0.5 rounded-md text-[11px] font-medium transition-colors cursor-pointer"
            :class="
              codeTab === 'builder'
                ? 'bg-primary text-white font-semibold'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
            "
          >
            Fluent Builder
          </button>
          <button
            type="button"
            @click="codeTab = 'options'"
            class="px-2 py-0.5 rounded-md text-[11px] font-medium transition-colors cursor-pointer"
            :class="
              codeTab === 'options'
                ? 'bg-primary text-white font-semibold'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
            "
          >
            Options
          </button>
          <button
            type="button"
            @click="codeTab = 'json'"
            class="px-2 py-0.5 rounded-md text-[11px] font-medium transition-colors cursor-pointer"
            :class="
              codeTab === 'json'
                ? 'bg-primary text-white font-semibold'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
            "
          >
            JSON
          </button>
        </div>

        <button
          type="button"
          @click="copyActiveCode"
          class="text-[11px] text-primary hover:underline font-medium cursor-pointer"
        >
          {{ copied ? '✓ Copied' : 'Copy snippet' }}
        </button>
      </div>

      <pre
        class="whitespace-pre overflow-x-auto text-zinc-800 dark:text-zinc-200 p-3 bg-zinc-50 dark:bg-zinc-950/80 font-mono text-[11px] leading-relaxed select-text"
        >{{ activeCodeSnippet }}</pre>
    </div>

    <div
      class="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-950/40 p-3 grid grid-cols-2 gap-x-4 gap-y-2 text-[11px]"
    >
      <div class="flex justify-between gap-2">
        <span class="text-zinc-500">Model</span>
        <span class="font-medium text-zinc-800 dark:text-zinc-200 truncate">{{
          displayConfig.modelName
        }}</span>
      </div>
      <div class="flex justify-between gap-2">
        <span class="text-zinc-500">Backend</span>
        <span class="font-mono font-medium text-zinc-800 dark:text-zinc-200">
          {{ displayConfig.backendLabel }}
        </span>
      </div>
      <div class="flex justify-between gap-2">
        <span class="text-zinc-500">Detector Limit</span>
        <span class="font-mono font-medium text-zinc-800 dark:text-zinc-200 tabular-nums">
          {{ displayConfig.options.detection?.limitSideLen ?? 736 }}px
        </span>
      </div>
      <div class="flex justify-between gap-2">
        <span class="text-zinc-500">Score Cutoff</span>
        <span class="font-mono font-medium text-zinc-800 dark:text-zinc-200 tabular-nums">
          {{ Math.round((displayConfig.options.postRecognition?.scoreThreshold ?? 0.5) * 100) }}%
        </span>
      </div>
      <div class="flex justify-between gap-2">
        <span class="text-zinc-500">Reading Order</span>
        <span class="font-mono font-medium text-zinc-800 dark:text-zinc-200">
          {{ displayConfig.options.layoutReading ? 'Enabled' : 'Off' }}
        </span>
      </div>
      <div class="flex justify-between gap-2">
        <span class="text-zinc-500">Auto-Unwarp</span>
        <span class="font-mono font-medium text-zinc-800 dark:text-zinc-200">
          {{ displayConfig.options.autoUnwarp ? 'Enabled' : 'Off' }}
        </span>
      </div>
    </div>
  </div>
</template>
