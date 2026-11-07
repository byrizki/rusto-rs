<script setup lang="ts">
import type { OcrConfig } from '../../../types/ocrConfig';

defineProps<{
  config: OcrConfig;
}>();

const sidePresets = [640, 960, 1280, 1920] as const;
</script>

<template>
  <div class="space-y-4 text-xs">
    <!-- Phase 2: Detection -->
    <div
      class="flex items-center justify-between pb-1 border-b border-zinc-100 dark:border-zinc-800"
    >
      <span class="text-[11px] font-semibold text-zinc-900 dark:text-zinc-100">
        Phase 2: Detection Inference
      </span>
      <span
        class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium"
      >
        .withDetection()
      </span>
    </div>

    <div
      class="space-y-2.5 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <div class="flex items-center justify-between">
        <label class="font-medium text-zinc-800 dark:text-zinc-200">
          Detector Input Limit
          <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1">(limitSideLen)</span>
        </label>
        <div class="flex items-center gap-1.5">
          <input
            v-model.number="config.detectionLimitSideLen"
            type="number"
            min="320"
            max="3840"
            step="160"
            class="w-20 px-2 py-1 text-xs font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-right tabular-nums"
          />
          <select
            v-model="config.detectionLimitType"
            class="px-2 py-1 text-xs rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 cursor-pointer"
          >
            <option value="max">Max</option>
            <option value="min">Min</option>
          </select>
        </div>
      </div>

      <div class="flex items-center gap-1.5 flex-wrap">
        <button
          v-for="side in sidePresets"
          :key="side"
          type="button"
          @click="config.detectionLimitSideLen = side"
          class="px-2 py-0.5 text-[11px] font-mono rounded-md border transition-colors cursor-pointer tabular-nums"
          :class="
            config.detectionLimitSideLen === side
              ? 'bg-primary text-white border-primary font-semibold'
              : 'border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400 hover:border-zinc-400'
          "
        >
          {{ side }}px
        </button>
      </div>
      <p class="text-[11px] text-zinc-500 dark:text-zinc-400">
        Constrains the image dimensions fed into the DBNet detection neural model.
      </p>
    </div>

    <!-- Phase 3: Post-Detection -->
    <div
      class="flex items-center justify-between pt-2 pb-1 border-b border-zinc-100 dark:border-zinc-800"
    >
      <span class="text-[11px] font-semibold text-zinc-900 dark:text-zinc-100">
        Phase 3: Post-Detection Filtering
      </span>
      <span
        class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium"
      >
        .withPostDetection()
      </span>
    </div>

    <div
      class="space-y-3 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        DBNet Contours & Thresholding
      </div>

      <div class="space-y-1">
        <div class="flex justify-between items-center">
          <label class="font-medium text-zinc-700 dark:text-zinc-300">
            Pixel Binarization
            <span class="font-mono text-[10px] text-zinc-400 font-normal ml-1">(threshold)</span>
          </label>
          <span
            class="font-mono text-[11px] text-zinc-700 dark:text-zinc-300 font-semibold tabular-nums"
          >
            {{ config.postprocessThreshold.toFixed(2) }}
          </span>
        </div>
        <input
          v-model.number="config.postprocessThreshold"
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          class="w-full accent-primary cursor-pointer"
        />
      </div>

      <div class="space-y-1">
        <div class="flex justify-between items-center">
          <label class="font-medium text-zinc-700 dark:text-zinc-300">
            Box Confidence
            <span class="font-mono text-[10px] text-zinc-400 font-normal ml-1">(boxThreshold)</span>
          </label>
          <span
            class="font-mono text-[11px] text-zinc-700 dark:text-zinc-300 font-semibold tabular-nums"
          >
            {{ config.postprocessBoxThreshold.toFixed(2) }}
          </span>
        </div>
        <input
          v-model.number="config.postprocessBoxThreshold"
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          class="w-full accent-primary cursor-pointer"
        />
      </div>

      <div class="space-y-1">
        <div class="flex justify-between items-center">
          <label class="font-medium text-zinc-700 dark:text-zinc-300">
            Polygon Expansion
            <span class="font-mono text-[10px] text-zinc-400 font-normal ml-1">(unclipRatio)</span>
          </label>
          <span
            class="font-mono text-[11px] text-zinc-700 dark:text-zinc-300 font-semibold tabular-nums"
          >
            {{ config.postprocessUnclipRatio.toFixed(1) }}
          </span>
        </div>
        <input
          v-model.number="config.postprocessUnclipRatio"
          type="range"
          min="1"
          max="3"
          step="0.1"
          class="w-full accent-primary cursor-pointer"
        />
      </div>

      <div
        class="flex items-center justify-between gap-3 pt-2 border-t border-zinc-200/60 dark:border-zinc-800"
      >
        <div class="flex items-center gap-2">
          <label class="text-[11px] font-medium text-zinc-600 dark:text-zinc-400">
            Max Candidates
          </label>
          <input
            v-model.number="config.postprocessMaxCandidates"
            type="number"
            min="100"
            max="3000"
            step="100"
            class="w-20 px-2 py-1 text-xs font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums"
          />
        </div>

        <label class="flex items-center gap-2 cursor-pointer">
          <input
            v-model="config.postprocessUseDilation"
            type="checkbox"
            class="rounded accent-primary w-4 h-4 cursor-pointer"
          />
          <span class="font-medium text-zinc-700 dark:text-zinc-300">Morphological Dilation</span>
        </label>
      </div>
    </div>

    <!-- Box Geometry Filtering -->
    <div
      class="space-y-2.5 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <label class="flex items-center justify-between cursor-pointer">
        <div>
          <span class="font-medium text-zinc-800 dark:text-zinc-200 block">
            Custom Bounding Box Geometry Filters
          </span>
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
            Filter out candidate boxes outside specific pixel dimensions or ratios
          </span>
        </div>
        <input
          v-model="config.enableResize"
          type="checkbox"
          class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
        />
      </label>

      <div
        v-if="config.enableResize"
        class="grid grid-cols-2 gap-2.5 pt-2 border-t border-zinc-200/60 dark:border-zinc-800"
      >
        <div class="space-y-1">
          <label class="text-[11px] text-zinc-500">Max Side (px)</label>
          <input
            v-model.number="config.maxSideLen"
            type="number"
            min="320"
            max="4096"
            step="128"
            class="w-full px-2 py-1 text-xs font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums"
          />
        </div>
        <div class="space-y-1">
          <label class="text-[11px] text-zinc-500">Min Side (px)</label>
          <input
            v-model.number="config.minSideLen"
            type="number"
            min="100"
            max="2048"
            step="64"
            class="w-full px-2 py-1 text-xs font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums"
          />
        </div>
        <div class="space-y-1">
          <label class="text-[11px] text-zinc-500">Min Height (px)</label>
          <input
            v-model.number="config.minHeight"
            type="number"
            min="16"
            max="512"
            step="16"
            class="w-full px-2 py-1 text-xs font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums"
          />
        </div>
        <div class="space-y-1">
          <label class="text-[11px] text-zinc-500">Aspect Ratio (-1 auto)</label>
          <input
            v-model.number="config.widthHeightRatio"
            type="number"
            min="-1"
            max="10"
            step="0.1"
            class="w-full px-2 py-1 text-xs font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums"
          />
        </div>
      </div>
    </div>
  </div>
</template>
