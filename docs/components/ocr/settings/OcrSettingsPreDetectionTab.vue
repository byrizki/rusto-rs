<script setup lang="ts">
import type { OcrConfig } from '../../../types/ocrConfig';

defineProps<{
  config: OcrConfig;
}>();
</script>

<template>
  <div class="space-y-4 text-xs">
    <!-- Phase 1 Badge -->
    <div
      class="flex items-center justify-between pb-1 border-b border-zinc-100 dark:border-zinc-800"
    >
      <span class="text-[11px] font-semibold text-zinc-900 dark:text-zinc-100">
        Phase 1: Pre-Detection
      </span>
      <span
        class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium"
      >
        .withPreDetection()
      </span>
    </div>

    <!-- Document Transforms -->
    <div
      class="space-y-3 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        Document Transforms & Rectification
      </div>

      <label class="flex items-center justify-between gap-3 cursor-pointer">
        <div>
          <span class="font-medium text-zinc-800 dark:text-zinc-200 block">
            3D Document Auto-Unwarp
            <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1"
              >.withAutoUnwarp()</span
            >
          </span>
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
            Flattens curved book pages, crumpled paper, and folded receipts
          </span>
        </div>
        <input
          v-model="config.autoUnwarp"
          type="checkbox"
          class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
        />
      </label>

      <label
        class="flex items-center justify-between gap-3 cursor-pointer pt-2 border-t border-zinc-200/60 dark:border-zinc-800"
      >
        <div>
          <span class="font-medium text-zinc-800 dark:text-zinc-200 block">
            Page Orientation Correction
            <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1"
              >.withOrientationDetection()</span
            >
          </span>
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
            Automatically detects and rotates 90°, 180°, and 270° rotated documents
          </span>
        </div>
        <input
          v-model="config.orientationDetection"
          type="checkbox"
          class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
        />
      </label>
    </div>

    <!-- Image Calibration & Descreening -->
    <div
      class="space-y-3 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        Illumination & Descreening Calibration
      </div>

      <label class="flex items-center justify-between gap-3 cursor-pointer">
        <div>
          <span class="font-medium text-zinc-800 dark:text-zinc-200 block">
            Adaptive Lighting Calibration
          </span>
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
            Normalizes harsh uneven illumination, flash glare, and shadow gradients
          </span>
        </div>
        <input
          v-model="config.calibrationEnabled"
          type="checkbox"
          class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
        />
      </label>

      <label
        class="flex items-center justify-between gap-3 cursor-pointer pt-2 border-t border-zinc-200/60 dark:border-zinc-800"
      >
        <div>
          <span class="font-medium text-zinc-800 dark:text-zinc-200 block">
            Descreen Patterned Backgrounds
          </span>
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
            Suppresses fine guilloche security patterns and halftone dots on IDs
          </span>
        </div>
        <input
          v-model="config.descreenBackground"
          type="checkbox"
          class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
        />
      </label>

      <div
        class="space-y-1 pt-1 transition-opacity"
        :class="{
          'opacity-40 pointer-events-none':
            !config.calibrationEnabled || !config.descreenBackground,
        }"
      >
        <div class="flex justify-between items-center">
          <label class="text-[11px] text-zinc-600 dark:text-zinc-400">
            Descreen Filtering Strength
            <span class="font-mono text-[10px] text-zinc-400">(descreenStrength)</span>
          </label>
          <span
            class="font-mono text-[11px] text-zinc-700 dark:text-zinc-300 font-semibold tabular-nums"
          >
            {{ config.descreenStrength.toFixed(2) }}
          </span>
        </div>
        <input
          v-model.number="config.descreenStrength"
          type="range"
          min="0.1"
          max="1.5"
          step="0.05"
          class="w-full accent-primary cursor-pointer"
        />
      </div>
    </div>

    <!-- SIMD Optimization & ROI Padding -->
    <div
      class="space-y-3 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        Resolution Scaling & Crop Padding
      </div>

      <label class="flex items-center justify-between gap-3 cursor-pointer">
        <div>
          <span class="font-medium text-zinc-800 dark:text-zinc-200 block">
            SIMD Fast Downscale & Crop
          </span>
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
            Pre-scales high-resolution camera images to save memory and inference time
          </span>
        </div>
        <input
          v-model="config.optimizationEnabled"
          type="checkbox"
          class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
        />
      </label>

      <div
        class="grid grid-cols-2 gap-2.5 transition-opacity pt-1 border-t border-zinc-200/60 dark:border-zinc-800"
        :class="{ 'opacity-40 pointer-events-none': !config.optimizationEnabled }"
      >
        <div class="space-y-1">
          <label class="text-[11px] text-zinc-600 dark:text-zinc-400"> Target Max Side (px) </label>
          <input
            v-model.number="config.optimizationTargetMaxSide"
            type="number"
            min="640"
            max="2560"
            step="128"
            class="w-full px-2 py-1 text-xs font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums"
          />
        </div>
        <div class="space-y-1">
          <label class="text-[11px] text-zinc-600 dark:text-zinc-400"> Crop Padding X (px) </label>
          <input
            v-model.number="config.cropPaddingX"
            type="number"
            min="0"
            max="50"
            step="2"
            class="w-full px-2 py-1 text-xs font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums"
          />
        </div>
      </div>
    </div>
  </div>
</template>
