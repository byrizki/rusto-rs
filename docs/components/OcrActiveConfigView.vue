<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ActiveRunConfig, OcrConfig } from '../types/ocrConfig';
import { buildDetectTextOptions } from '../types/ocrConfig';

const props = defineProps<{
  activeRun: ActiveRunConfig | null;
  currentOcrConfig: OcrConfig;
  currentModelName: string;
  currentOutputMode: 'lines' | 'words';
  isModifiedSinceRun: boolean;
}>();

const copied = ref<boolean>(false);

const displayConfig = computed(() => {
  if (props.activeRun) {
    return {
      isSnapshot: true,
      modelName: props.activeRun.modelName,
      outputMode: props.activeRun.outputMode,
      durationMs: props.activeRun.durationMs,
      itemsCount: props.activeRun.itemsCount,
      options: props.activeRun.options,
      rawConfig: props.activeRun.rawConfig,
    };
  }
  return {
    isSnapshot: false,
    modelName: props.currentModelName,
    outputMode: props.currentOutputMode,
    durationMs: null,
    itemsCount: 0,
    options: buildDetectTextOptions(props.currentOcrConfig, props.currentOutputMode),
    rawConfig: props.currentOcrConfig,
  };
});

const jsonOptionsFormatted = computed(() => {
  return JSON.stringify(displayConfig.value.options, null, 2);
});

async function copyJson() {
  try {
    await navigator.clipboard.writeText(jsonOptionsFormatted.value);
    copied.value = true;
    setTimeout(() => {
      copied.value = false;
    }, 2000);
  } catch (err) {
    console.error('Copy options JSON failed:', err);
  }
}
</script>

<template>
  <div class="space-y-3.5 text-xs">
    <!-- Header Status & Sync Banner -->
    <div
      class="p-2.5 rounded-lg border flex flex-wrap items-center justify-between gap-2"
      :class="
        displayConfig.isSnapshot
          ? isModifiedSinceRun
            ? 'border-amber-200 dark:border-amber-900/60 bg-amber-50/70 dark:bg-amber-950/20'
            : 'border-emerald-200 dark:border-emerald-900/60 bg-emerald-50/70 dark:bg-emerald-950/20'
          : 'border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900'
      "
    >
      <div class="flex items-center gap-2">
        <span
          class="w-2 h-2 rounded-full shrink-0"
          :class="
            displayConfig.isSnapshot
              ? isModifiedSinceRun
                ? 'bg-amber-500'
                : 'bg-emerald-500'
              : 'bg-neutral-400'
          "
        />
        <span
          class="font-semibold"
          :class="
            displayConfig.isSnapshot
              ? isModifiedSinceRun
                ? 'text-amber-800 dark:text-amber-300'
                : 'text-emerald-800 dark:text-emerald-300'
              : 'text-neutral-700 dark:text-neutral-300'
          "
        >
          {{
            displayConfig.isSnapshot
              ? isModifiedSinceRun
                ? 'Active Run (Settings Modified)'
                : 'Active Run In Sync'
              : 'Pending Run (Current Defaults)'
          }}
        </span>
      </div>

      <div class="flex items-center gap-2 font-mono text-[11px]">
        <span
          v-if="displayConfig.durationMs !== null"
          class="text-neutral-600 dark:text-neutral-400"
        >
          {{ displayConfig.durationMs }} ms
        </span>
        <span
          class="px-2 py-0.5 rounded font-sans font-medium"
          :class="
            displayConfig.isSnapshot
              ? 'bg-neutral-200/70 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300'
              : 'bg-neutral-200/50 dark:bg-neutral-800 text-neutral-500'
          "
        >
          {{ displayConfig.isSnapshot ? `${displayConfig.itemsCount} detected` : 'Not run yet' }}
        </span>
      </div>
    </div>

    <!-- Alert when settings modified since last run -->
    <div
      v-if="displayConfig.isSnapshot && isModifiedSinceRun"
      class="text-[11px] text-amber-700 dark:text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-md px-2.5 py-1.5"
    >
      Parameters have been modified in OCR Engine Settings. Click <strong>Run OCR</strong> to
      execute with new parameters.
    </div>

    <!-- Active Run Overview Cards -->
    <div class="grid grid-cols-2 gap-2 font-sans">
      <div
        class="p-2 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50"
      >
        <span
          class="text-[10px] text-neutral-500 dark:text-neutral-400 uppercase tracking-wider block"
          >Model</span
        >
        <span class="font-semibold text-neutral-800 dark:text-neutral-200">{{
          displayConfig.modelName
        }}</span>
      </div>
      <div
        class="p-2 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50"
      >
        <span
          class="text-[10px] text-neutral-500 dark:text-neutral-400 uppercase tracking-wider block"
          >Granularity</span
        >
        <span class="font-semibold capitalize text-neutral-800 dark:text-neutral-200">{{
          displayConfig.outputMode
        }}</span>
      </div>
    </div>

    <!-- Section 1: Detection & Postprocess -->
    <div
      class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden font-sans"
    >
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 font-semibold text-[11px] text-neutral-700 dark:text-neutral-300"
      >
        Detection & Postprocess
      </div>
      <div class="p-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Min Confidence:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ Math.round((displayConfig.options.textScore ?? 0) * 100) }}%
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Limit Side Length:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.detection?.limitSideLen ?? 960 }}px ({{
              displayConfig.options.detection?.limitType ?? 'max'
            }})
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Box Confidence:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postprocess?.boxThreshold?.toFixed(2) ?? '0.50' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">DBNet Pixel Thresh:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postprocess?.threshold?.toFixed(2) ?? '0.30' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Box Unclip Ratio:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postprocess?.unclipRatio?.toFixed(1) ?? '1.6' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Max Candidates:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postprocess?.maxCandidates ?? 1000 }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Use Dilation:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postprocess?.useDilation ? 'Yes' : 'No' }}
          </span>
        </div>
      </div>
    </div>

    <!-- Section 2: Layout & Direction -->
    <div
      class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden font-sans"
    >
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 font-semibold text-[11px] text-neutral-700 dark:text-neutral-300"
      >
        Layout & Grouping
      </div>
      <div class="p-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Row Center (lineY):</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.lineYThreshold?.toFixed(2) ?? '0.50' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Word Gap (wordX):</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.wordXThreshold?.toFixed(2) ?? '0.40' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Orientation Classifier:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.classification ? 'Enabled' : 'Disabled' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">4-Way Direction:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.orientation ? 'Enabled' : 'Disabled' }}
          </span>
        </div>
      </div>
    </div>

    <!-- Section 3: Preprocess & Optimization -->
    <div
      class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden font-sans"
    >
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 font-semibold text-[11px] text-neutral-700 dark:text-neutral-300"
      >
        Preprocessing & Optimization
      </div>
      <div class="p-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Lighting Calibration:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.calibration?.enabled ? 'Enabled' : 'Disabled' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Descreen Background:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{
              displayConfig.options.calibration?.descreenBackground
                ? `Enabled (${displayConfig.options.calibration.descreenStrength?.toFixed(2) ?? '1.15'})`
                : 'Disabled'
            }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">SIMD / Crop Opt:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{
              displayConfig.options.optimization?.enabled
                ? `Enabled (${displayConfig.options.optimization.targetMaxSide ?? 1280}px)`
                : 'Disabled'
            }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Geometry Resizing:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.rawConfig.enableResize ? 'Custom' : 'Disabled' }}
          </span>
        </div>
      </div>
    </div>

    <!-- Section 4: Raw DetectTextOptions JSON -->
    <div class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden">
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 flex items-center justify-between font-sans"
      >
        <span class="font-semibold text-[11px] text-neutral-700 dark:text-neutral-300">
          Raw DetectTextOptions Object
        </span>
        <button
          type="button"
          @click="copyJson"
          class="text-[10px] text-primary hover:underline font-mono cursor-pointer"
        >
          {{ copied ? '✓ Copied' : 'Copy Options JSON' }}
        </button>
      </div>
      <pre
        class="whitespace-pre overflow-x-auto text-neutral-700 dark:text-neutral-300 p-2.5 bg-neutral-50 dark:bg-neutral-950 font-mono text-[11px] leading-relaxed select-text"
        style="
          font-family:
            ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New',
            monospace;
          tab-size: 2;
        "
        >{{ jsonOptionsFormatted }}</pre>
    </div>
  </div>
</template>
