<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ActiveRunConfig, OcrConfig } from '../types/ocrConfig';
import { buildDetectTextOptions } from '../types/ocrConfig';
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

const fluentBuilderCode = computed(() => {
  return formatFluentBuilderSnippet(displayConfig.value.outputMode, displayConfig.value.options);
});

const optionsObjectCode = computed(() => {
  return formatOptionsSnippet(displayConfig.value.options);
});

const jsonOptionsFormatted = computed(() => {
  return JSON.stringify(displayConfig.value.options, null, 2);
});

const activeCodeSnippet = computed(() => {
  if (codeTab.value === 'builder') return fluentBuilderCode.value;
  if (codeTab.value === 'options') return optionsObjectCode.value;
  return jsonOptionsFormatted.value;
});

async function copyActiveCode() {
  try {
    await navigator.clipboard.writeText(activeCodeSnippet.value);
    copied.value = true;
    setTimeout(() => {
      copied.value = false;
    }, 2000);
  } catch (err) {
    console.error('Copy options code failed:', err);
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
    <div class="grid grid-cols-3 gap-2 font-sans">
      <div
        class="p-2 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50"
      >
        <span
          class="text-[10px] text-neutral-500 dark:text-neutral-400 uppercase tracking-wider block"
          >Model</span
        >
        <span class="font-semibold text-neutral-800 dark:text-neutral-200 truncate block">{{
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
          displayConfig.outputMode === 'words' ? 'asWords()' : 'asLines()'
        }}</span>
      </div>
      <div
        class="p-2 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50"
      >
        <span
          class="text-[10px] text-neutral-500 dark:text-neutral-400 uppercase tracking-wider block"
          >Acceleration</span
        >
        <span
          class="font-semibold"
          :class="
            displayConfig.options.webgpu
              ? 'text-emerald-600 dark:text-emerald-400'
              : 'text-neutral-800 dark:text-neutral-200'
          "
        >
          {{ displayConfig.options.webgpu ? 'WebGPU' : 'WASM SIMD' }}
        </span>
      </div>
    </div>

    <!-- Section 1: Phase 1 & 2 - Pre-Detection & Detection -->
    <div
      class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden font-sans"
    >
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 font-semibold text-[11px] text-neutral-700 dark:text-neutral-300 flex items-center justify-between"
      >
        <span>Phase 1 & 2: Pre-Detection & Detection</span>
        <span class="text-[10px] text-neutral-500 font-mono">withPreDetection / withDetection</span>
      </div>
      <div class="p-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Limit Side Length:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.detection?.limitSideLen ?? 960 }}px ({{
              displayConfig.options.detection?.limitType ?? 'max'
            }})
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Descreen Background:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{
              displayConfig.options.preDetection?.descreenStrength !== undefined
                ? `Strength ${displayConfig.options.preDetection.descreenStrength.toFixed(2)}`
                : 'Disabled'
            }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Target Max Side:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{
              displayConfig.options.preDetection?.targetMaxSide !== undefined
                ? `${displayConfig.options.preDetection.targetMaxSide}px`
                : 'Default'
            }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Crop Padding (X/Y):</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.preDetection?.cropPaddingX ?? 0 }} /
            {{ displayConfig.options.preDetection?.cropPaddingY ?? 0 }}
          </span>
        </div>
      </div>
    </div>

    <!-- Section 2: Phase 3 - Post-Detection -->
    <div
      class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden font-sans"
    >
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 font-semibold text-[11px] text-neutral-700 dark:text-neutral-300 flex items-center justify-between"
      >
        <span>Phase 3: Post-Detection Filtering & Geometry</span>
        <span class="text-[10px] text-neutral-500 font-mono">withPostDetection</span>
      </div>
      <div class="p-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Box Confidence:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postDetection?.boxThreshold?.toFixed(2) ?? '0.50' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">DBNet Pixel Thresh:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postDetection?.threshold?.toFixed(2) ?? '0.30' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Box Unclip Ratio:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postDetection?.unclipRatio?.toFixed(1) ?? '1.8' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Max Candidates:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postDetection?.maxCandidates ?? 1000 }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Use Dilation:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.postDetection?.useDilation ? 'Yes' : 'No' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Geometry Bounds:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{
              displayConfig.rawConfig.enableResize
                ? `${displayConfig.rawConfig.minSideLen}-${displayConfig.rawConfig.maxSideLen}px`
                : 'Auto'
            }}
          </span>
        </div>
      </div>
    </div>

    <!-- Section 3: Phase 4, 5, 6 - Pre-Recognition, Recognition, Post-Recognition -->
    <div
      class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden font-sans"
    >
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 font-semibold text-[11px] text-neutral-700 dark:text-neutral-300 flex items-center justify-between"
      >
        <span>Phase 4, 5 & 6: Recognition & Grouping</span>
        <span class="text-[10px] text-neutral-500 font-mono"
          >withPreRecog / withRecog / withPostRecog</span
        >
      </div>
      <div class="p-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Row Center (lineY):</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.preRecognition?.lineYThreshold?.toFixed(2) ?? '0.50' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Word Gap (wordX):</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.preRecognition?.wordXThreshold?.toFixed(2) ?? '0.40' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Batch Size:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.recognition?.batchSize ?? 1 }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Min Score Cutoff:</span>
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ Math.round((displayConfig.options.postRecognition?.scoreThreshold ?? 0.5) * 100) }}%
          </span>
        </div>
      </div>
    </div>

    <!-- Section 4: Auxiliary Pipelines -->
    <div
      class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden font-sans"
    >
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 font-semibold text-[11px] text-neutral-700 dark:text-neutral-300"
      >
        Auxiliary Document Processing Pipelines
      </div>
      <div class="p-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400">Auto-Unwarp (withAutoUnwarp):</span>
          <span
            class="font-mono font-medium"
            :class="
              displayConfig.options.autoUnwarp
                ? 'text-emerald-600 dark:text-emerald-400 font-semibold'
                : 'text-neutral-800 dark:text-neutral-200'
            "
          >
            {{ displayConfig.options.autoUnwarp ? 'Enabled' : 'Disabled' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400"
            >Reading Order (withLayoutReading):</span
          >
          <span
            class="font-mono font-medium"
            :class="
              displayConfig.options.layoutReading
                ? 'text-emerald-600 dark:text-emerald-400 font-semibold'
                : 'text-neutral-800 dark:text-neutral-200'
            "
          >
            {{ displayConfig.options.layoutReading ? 'Natural Column Flow' : 'Disabled' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400"
            >Line Classifier (withLineClassification):</span
          >
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.lineClassification ? 'Enabled (0°/180°)' : 'Disabled' }}
          </span>
        </div>
        <div class="flex justify-between items-center">
          <span class="text-neutral-500 dark:text-neutral-400"
            >Orientation (withOrientationDetection):</span
          >
          <span class="font-mono font-medium text-neutral-800 dark:text-neutral-200">
            {{ displayConfig.options.orientationDetection ? 'Enabled (4-Way)' : 'Disabled' }}
          </span>
        </div>
      </div>
    </div>

    <!-- Section 5: Code Snippet Generation -->
    <div class="rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden">
      <div
        class="px-3 py-1.5 bg-neutral-100/70 dark:bg-neutral-800/60 flex items-center justify-between font-sans"
      >
        <div class="flex items-center gap-1.5">
          <button
            type="button"
            @click="codeTab = 'builder'"
            class="px-2 py-0.5 rounded text-[10px] font-medium transition-colors cursor-pointer"
            :class="
              codeTab === 'builder'
                ? 'bg-primary text-white font-semibold'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
            "
          >
            Fluent Builder API
          </button>
          <button
            type="button"
            @click="codeTab = 'options'"
            class="px-2 py-0.5 rounded text-[10px] font-medium transition-colors cursor-pointer"
            :class="
              codeTab === 'options'
                ? 'bg-primary text-white font-semibold'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
            "
          >
            OcrRunOptions
          </button>
          <button
            type="button"
            @click="codeTab = 'json'"
            class="px-2 py-0.5 rounded text-[10px] font-medium transition-colors cursor-pointer"
            :class="
              codeTab === 'json'
                ? 'bg-primary text-white font-semibold'
                : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
            "
          >
            Raw JSON
          </button>
        </div>
        <button
          type="button"
          @click="copyActiveCode"
          class="text-[10px] text-primary hover:underline font-mono cursor-pointer"
        >
          {{ copied ? '✓ Copied' : 'Copy Code' }}
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
        >{{ activeCodeSnippet }}</pre>
    </div>
  </div>
</template>
