<script setup lang="ts">
import { computed, ref } from 'vue';
import type { OcrConfig } from '../../../types/ocrConfig';
import OcrInfoPopover from './OcrInfoPopover.vue';
import OcrSettingsCollapsibleSection from './OcrSettingsCollapsibleSection.vue';

const props = defineProps<{
  config: OcrConfig;
}>();

const sidePresets = [640, 960, 1280, 1920] as const;

type DetectionSection = 'pre' | 'core' | 'post';
const activeSection = ref<DetectionSection | null>('core');

function onToggle(section: DetectionSection, nextValue: boolean) {
  if (nextValue) {
    activeSection.value = section;
  } else if (activeSection.value === section) {
    activeSection.value = null;
  }
}

const preDetectionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.autoUnwarp) count++;
  if (props.config.orientationDetection) count++;
  if (props.config.descreenBackground) count++;
  if (props.config.optimizationEnabled) count++;
  return count;
});

const detectionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.detectionLimitSideLen !== 960) count++;
  if (props.config.detectionLimitType !== 'max') count++;
  return count;
});

const postDetectionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.enableResize) count++;
  if (props.config.postprocessUseDilation) count++;
  if (props.config.postprocessThreshold !== 0.3) count++;
  if (props.config.postprocessBoxThreshold !== 0.5) count++;
  return count;
});
</script>

<template>
  <div class="space-y-3 text-xs">
    <!-- Phase 1: Pre-Detection (Collapsible) -->
    <OcrSettingsCollapsibleSection
      :model-value="activeSection === 'pre'"
      @update:model-value="(val) => onToggle('pre', val)"
      title="Phase 1: Pre-Detection"
      badge=".withPreDetection()"
      description="Document auto-unwarp, orientation, descreening & downscaling"
      :active-count="preDetectionActiveCount"
    >
      <!-- Document Transforms -->
      <div class="space-y-2.5 pt-1">
        <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
          Document Transforms & Rectification
        </div>

        <label class="flex items-center justify-between gap-3 cursor-pointer">
          <div class="flex items-center gap-1.5 flex-wrap">
            <span class="font-medium text-zinc-800 dark:text-zinc-200">
              3D Document Auto-Unwarp
            </span>
            <span class="text-[10px] font-mono text-zinc-400 font-normal">.withAutoUnwarp()</span>
            <OcrInfoPopover
              text="Flattens curved book pages, crumpled paper, and folded receipts using 3D spatial unwarp meshes."
            />
          </div>
          <input
            v-model="config.autoUnwarp"
            type="checkbox"
            class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
          />
        </label>

        <label
          class="flex items-center justify-between gap-3 cursor-pointer pt-2 border-t border-zinc-100 dark:border-zinc-800"
        >
          <div class="flex items-center gap-1.5 flex-wrap">
            <span class="font-medium text-zinc-800 dark:text-zinc-200">
              Page Orientation Correction
            </span>
            <span class="text-[10px] font-mono text-zinc-400 font-normal"
              >.withOrientationDetection()</span
            >
            <OcrInfoPopover
              text="Automatically detects and rotates 90°, 180°, and 270° rotated documents."
            />
          </div>
          <input
            v-model="config.orientationDetection"
            type="checkbox"
            class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
          />
        </label>
      </div>

      <!-- Illumination & Descreening Calibration -->
      <div class="space-y-2.5 pt-2 border-t border-zinc-100 dark:border-zinc-800">
        <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
          Illumination & Descreening Calibration
        </div>

        <label class="flex items-center justify-between gap-3 cursor-pointer">
          <div class="flex items-center gap-1.5">
            <span class="font-medium text-zinc-800 dark:text-zinc-200">
              Adaptive Lighting Calibration
            </span>
            <OcrInfoPopover
              text="Normalizes harsh uneven illumination, flash glare, and shadow gradients across document surfaces."
            />
          </div>
          <input
            v-model="config.calibrationEnabled"
            type="checkbox"
            class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
          />
        </label>

        <label
          class="flex items-center justify-between gap-3 cursor-pointer pt-2 border-t border-zinc-100 dark:border-zinc-800"
        >
          <div class="flex items-center gap-1.5">
            <span class="font-medium text-zinc-800 dark:text-zinc-200">
              Descreen Patterned Backgrounds
            </span>
            <OcrInfoPopover
              text="Suppresses fine guilloche security patterns and halftone dots on ID cards and passports."
            />
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
            <div class="flex items-center gap-1.5">
              <label class="text-[11px] text-zinc-600 dark:text-zinc-400">
                Descreen Filtering Strength
              </label>
              <OcrInfoPopover
                text="Controls the intensity of Gaussian & median descreening passes."
              />
            </div>
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

      <!-- Resolution Scaling & Crop Padding -->
      <div class="space-y-2.5 pt-2 border-t border-zinc-100 dark:border-zinc-800">
        <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
          Resolution Scaling & Crop Padding
        </div>

        <label class="flex items-center justify-between gap-3 cursor-pointer">
          <div class="flex items-center gap-1.5">
            <span class="font-medium text-zinc-800 dark:text-zinc-200">
              SIMD Fast Downscale & Crop
            </span>
            <OcrInfoPopover
              text="Pre-scales high-resolution camera photos to save memory and inference time."
            />
          </div>
          <input
            v-model="config.optimizationEnabled"
            type="checkbox"
            class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
          />
        </label>

        <div
          class="grid grid-cols-2 gap-2.5 transition-opacity pt-1 border-t border-zinc-100 dark:border-zinc-800"
          :class="{ 'opacity-40 pointer-events-none': !config.optimizationEnabled }"
        >
          <div class="space-y-1">
            <label class="text-[11px] text-zinc-600 dark:text-zinc-400">
              Target Max Side (px)
            </label>
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
            <label class="text-[11px] text-zinc-600 dark:text-zinc-400">
              Crop Padding X (px)
            </label>
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
    </OcrSettingsCollapsibleSection>

    <!-- Phase 2: Core Text Detection (Collapsible) -->
    <OcrSettingsCollapsibleSection
      :model-value="activeSection === 'core'"
      @update:model-value="(val) => onToggle('core', val)"
      title="Phase 2: Text Detection"
      badge=".withDetection()"
      description="Input resolution limit (limitSideLen) & detection constraints"
      :active-count="detectionActiveCount"
    >
      <div class="space-y-2.5 pt-1">
        <div class="flex items-center justify-between gap-2">
          <div class="flex items-center gap-1.5">
            <label class="font-medium text-zinc-800 dark:text-zinc-200">
              Detector Input Limit
              <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1"
                >(limitSideLen)</span
              >
            </label>
            <OcrInfoPopover
              text="Constrains input dimensions fed into DBNet detection model. Higher values detect smaller text at higher memory cost."
            />
          </div>
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
      </div>
    </OcrSettingsCollapsibleSection>

    <!-- Phase 3: Post-Detection Filtering (Collapsible) -->
    <OcrSettingsCollapsibleSection
      :model-value="activeSection === 'post'"
      @update:model-value="(val) => onToggle('post', val)"
      title="Phase 3: Post-Detection Filtering"
      badge=".withPostDetection()"
      description="DBNet binarization, box confidence, unclip ratio & geometry filters"
      :active-count="postDetectionActiveCount"
    >
      <div class="space-y-3 pt-1">
        <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
          DBNet Contours & Thresholding
        </div>

        <div class="space-y-1">
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-1.5">
              <label class="font-medium text-zinc-700 dark:text-zinc-300">
                Pixel Binarization
                <span class="font-mono text-[10px] text-zinc-400 font-normal ml-1"
                  >(threshold)</span
                >
              </label>
              <OcrInfoPopover
                text="Probability cutoff for classifying a pixel as text vs background during DBNet binarization."
              />
            </div>
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
            <div class="flex items-center gap-1.5">
              <label class="font-medium text-zinc-700 dark:text-zinc-300">
                Box Confidence
                <span class="font-mono text-[10px] text-zinc-400 font-normal ml-1"
                  >(boxThreshold)</span
                >
              </label>
              <OcrInfoPopover
                text="Minimum average score across a detected polygon required to accept it as a valid text box."
              />
            </div>
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
            <div class="flex items-center gap-1.5">
              <label class="font-medium text-zinc-700 dark:text-zinc-300">
                Polygon Expansion
                <span class="font-mono text-[10px] text-zinc-400 font-normal ml-1"
                  >(unclipRatio)</span
                >
              </label>
              <OcrInfoPopover
                text="Vatti clipping expansion factor used to restore shrunk DBNet probability regions back to original text boundaries."
              />
            </div>
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
          class="flex items-center justify-between gap-3 pt-2 border-t border-zinc-100 dark:border-zinc-800"
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
            <OcrInfoPopover
              text="Applies 2x2 morphological dilation to connect broken character fragments."
            />
          </label>
        </div>
      </div>

      <!-- Box Geometry Filtering -->
      <div class="space-y-2.5 pt-2 border-t border-zinc-100 dark:border-zinc-800">
        <label class="flex items-center justify-between cursor-pointer">
          <div class="flex items-center gap-1.5">
            <span class="font-medium text-zinc-800 dark:text-zinc-200">
              Custom Bounding Box Geometry Filters
            </span>
            <OcrInfoPopover
              text="Filter out candidate boxes outside specific pixel dimensions or aspect ratios."
            />
          </div>
          <input
            v-model="config.enableResize"
            type="checkbox"
            class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
          />
        </label>

        <div
          v-if="config.enableResize"
          class="grid grid-cols-2 gap-2.5 pt-2 border-t border-zinc-100 dark:border-zinc-800"
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
    </OcrSettingsCollapsibleSection>
  </div>
</template>
