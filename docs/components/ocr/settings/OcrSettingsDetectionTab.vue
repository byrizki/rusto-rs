<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { defaultOcrConfig, type OcrConfig } from '../../../types/ocrConfig';
import OcrRangeParam from './OcrRangeParam.vue';
import OcrSettingRow from './OcrSettingRow.vue';
import OcrSettingsSubTabs, { type SettingsSubTab } from './OcrSettingsSubTabs.vue';

type DetectionSection = 'pre' | 'core' | 'post';

const props = defineProps<{
  config: OcrConfig;
  subTab?: DetectionSection;
}>();

const emit = defineEmits<{
  'update:subTab': [tab: DetectionSection];
}>();

const sidePresets = [640, 736, 960, 1536, 1920] as const;

const activeSection = ref<DetectionSection>(props.subTab ?? 'core');

watch(
  () => props.subTab,
  (val) => {
    if (val) activeSection.value = val;
  }
);

watch(activeSection, (val) => {
  emit('update:subTab', val);
});

const preDetectionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.autoUnwarp) count++;
  if (props.config.orientationDetection) count++;
  if (props.config.calibrationEnabled) {
    count++;
    if (
      props.config.descreenBackground &&
      props.config.descreenStrength !== defaultOcrConfig.descreenStrength
    ) {
      count++;
    }
  }
  if (props.config.optimizationEnabled) {
    count++;
    if (props.config.cropPaddingX !== defaultOcrConfig.cropPaddingX) count++;
    if (props.config.cropPaddingY !== defaultOcrConfig.cropPaddingY) count++;
  }
  return count;
});

const detectionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.detectionLimitSideLen !== defaultOcrConfig.detectionLimitSideLen) count++;
  if (props.config.detectionLimitType !== defaultOcrConfig.detectionLimitType) count++;
  return count;
});

const postDetectionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.enableResize) count++;
  if (props.config.postprocessUseDilation !== defaultOcrConfig.postprocessUseDilation) count++;
  if (props.config.postprocessThreshold !== defaultOcrConfig.postprocessThreshold) count++;
  if (props.config.postprocessBoxThreshold !== defaultOcrConfig.postprocessBoxThreshold) count++;
  if (props.config.postprocessUnclipRatio !== defaultOcrConfig.postprocessUnclipRatio) count++;
  if (props.config.postprocessMaxCandidates !== defaultOcrConfig.postprocessMaxCandidates) count++;
  return count;
});

const subTabs = computed<SettingsSubTab[]>(() => [
  {
    id: 'pre',
    label: 'Pre-Detection',
    badge: '.withPreDetection()',
    description: 'Document auto-unwarp, orientation, descreening & downscaling',
    activeCount: preDetectionActiveCount.value,
  },
  {
    id: 'core',
    label: 'Text Detection',
    badge: '.withDetection()',
    description: 'Input resolution limit (limitSideLen) & detection constraints',
    activeCount: detectionActiveCount.value,
  },
  {
    id: 'post',
    label: 'Post-Detection',
    badge: '.withPostDetection()',
    description: 'DBNet binarization, box confidence, unclip ratio & geometry filters',
    activeCount: postDetectionActiveCount.value,
  },
]);

function resetGroup(group: 'pre' | 'core' | 'post') {
  if (group === 'pre') {
    props.config.autoUnwarp = defaultOcrConfig.autoUnwarp;
    props.config.orientationDetection = defaultOcrConfig.orientationDetection;
    props.config.calibrationEnabled = defaultOcrConfig.calibrationEnabled;
    props.config.descreenBackground = defaultOcrConfig.descreenBackground;
    props.config.descreenStrength = defaultOcrConfig.descreenStrength;
    props.config.optimizationEnabled = defaultOcrConfig.optimizationEnabled;
    props.config.optimizationTargetMaxSide = defaultOcrConfig.optimizationTargetMaxSide;
    props.config.cropPaddingX = defaultOcrConfig.cropPaddingX;
    props.config.cropPaddingY = defaultOcrConfig.cropPaddingY;
  } else if (group === 'core') {
    props.config.detectionLimitSideLen = defaultOcrConfig.detectionLimitSideLen;
    props.config.detectionLimitType = defaultOcrConfig.detectionLimitType;
  } else {
    props.config.postprocessThreshold = defaultOcrConfig.postprocessThreshold;
    props.config.postprocessBoxThreshold = defaultOcrConfig.postprocessBoxThreshold;
    props.config.postprocessUnclipRatio = defaultOcrConfig.postprocessUnclipRatio;
    props.config.postprocessMaxCandidates = defaultOcrConfig.postprocessMaxCandidates;
    props.config.postprocessUseDilation = defaultOcrConfig.postprocessUseDilation;
    props.config.enableResize = defaultOcrConfig.enableResize;
    props.config.maxSideLen = defaultOcrConfig.maxSideLen;
    props.config.minSideLen = defaultOcrConfig.minSideLen;
    props.config.minHeight = defaultOcrConfig.minHeight;
    props.config.widthHeightRatio = defaultOcrConfig.widthHeightRatio;
  }
}
</script>

<template>
  <OcrSettingsSubTabs v-model="activeSection" :tabs="subTabs" class="text-xs">
    <!-- Phase 1: Pre-Detection -->
    <template v-if="activeSection === 'pre'">
      <div class="space-y-3">
        <!-- Document Transforms & Rectification -->
        <div>
          <div class="flex items-center justify-between mb-1">
            <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
              Document Transforms &amp; Rectification
            </span>
            <button
              v-if="preDetectionActiveCount > 0"
              type="button"
              @click="resetGroup('pre')"
              class="text-[10px] font-mono text-zinc-400 hover:text-primary cursor-pointer"
            >
              Reset Pre-Detection
            </button>
          </div>

          <OcrSettingRow
            setting-id="autoUnwarp"
            label="3D Document Auto-Unwarp"
            identifier=".withAutoUnwarp()"
            description="Flattens curved pages, crumpled paper, and folded receipts via 3D UVDoc mesh."
            tooltip="Runs UVDoc geometric unwarping prior to DBNet text detection. Enabling downloads an extra uvdoc.mnn neural model (+31 MB)."
            model-badge="+31 MB model"
            model-help="Adds ~31 MB UVDoc neural model (uvdoc.mnn) download when enabled."
            :is-modified="config.autoUnwarp !== defaultOcrConfig.autoUnwarp"
          >
            <input
              v-model="config.autoUnwarp"
              type="checkbox"
              aria-label="3D Document Auto-Unwarp"
              class="rounded accent-primary w-4 h-4 cursor-pointer"
            />
          </OcrSettingRow>

          <OcrSettingRow
            setting-id="orientationDetection"
            label="Page Orientation Correction"
            identifier=".withOrientationDetection()"
            description="Detects and rotates 90°, 180°, and 270° rotated documents."
            tooltip="Classifies full-page orientation before text detection. Enabling downloads an extra orient.mnn neural classifier (+6.5 MB)."
            model-badge="+6.5 MB model"
            model-help="Adds ~6.5 MB page orientation neural model (orient.mnn) download when enabled."
            :is-modified="config.orientationDetection !== defaultOcrConfig.orientationDetection"
          >
            <input
              v-model="config.orientationDetection"
              type="checkbox"
              aria-label="Page Orientation Correction"
              class="rounded accent-primary w-4 h-4 cursor-pointer"
            />
          </OcrSettingRow>
        </div>

        <!-- Illumination & Descreening Calibration -->
        <div class="pt-2 border-t border-zinc-200/70 dark:border-zinc-800">
          <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-1">
            Illumination &amp; Descreening Calibration
          </div>

          <OcrSettingRow
            setting-id="calibrationEnabled"
            label="Adaptive Lighting Calibration"
            identifier="calibrationEnabled"
            description="Normalizes uneven illumination, flash glare, and shadow gradients."
            tooltip="Enables pre-detection illumination normalization and optional descreening."
            :is-modified="config.calibrationEnabled !== defaultOcrConfig.calibrationEnabled"
          >
            <input
              v-model="config.calibrationEnabled"
              type="checkbox"
              aria-label="Adaptive Lighting Calibration"
              class="rounded accent-primary w-4 h-4 cursor-pointer"
            />
          </OcrSettingRow>

          <OcrSettingRow
            setting-id="descreenBackground"
            label="Descreen Patterned Backgrounds"
            identifier="descreenBackground"
            description="Suppresses guilloche security patterns and halftone dots on IDs & passports."
            tooltip="Applies Gaussian & median descreening passes when calibration is active."
            :disabled="!config.calibrationEnabled"
            disabled-reason="Requires Adaptive Lighting Calibration"
            :is-modified="config.descreenBackground !== defaultOcrConfig.descreenBackground"
          >
            <input
              v-model="config.descreenBackground"
              type="checkbox"
              :disabled="!config.calibrationEnabled"
              aria-label="Descreen Patterned Backgrounds"
              class="rounded accent-primary w-4 h-4 cursor-pointer disabled:cursor-not-allowed"
            />
          </OcrSettingRow>

          <OcrRangeParam
            setting-id="descreenStrength"
            v-model="config.descreenStrength"
            label="Descreen Filtering Strength"
            identifier="descreenStrength"
            tooltip="Controls intensity of Gaussian & median descreening passes."
            :min="0.1"
            :max="1.5"
            :step="0.05"
            :decimals="2"
            :default-value="defaultOcrConfig.descreenStrength"
            :disabled="!config.calibrationEnabled || !config.descreenBackground"
            disabled-reason="Requires Calibration and Descreen enabled"
          />
        </div>

        <!-- Resolution Scaling & Crop Padding -->
        <div class="pt-2 border-t border-zinc-200/70 dark:border-zinc-800">
          <div class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-1">
            Resolution Scaling &amp; Crop Padding
          </div>

          <OcrSettingRow
            setting-id="optimizationEnabled"
            label="SIMD Fast Downscale &amp; Crop"
            identifier="optimizationEnabled"
            description="Pre-scales high-resolution camera photos to reduce memory and latency."
            tooltip="Downscales input images whose longest side exceeds targetMaxSide before detection."
            :is-modified="config.optimizationEnabled !== defaultOcrConfig.optimizationEnabled"
          >
            <input
              v-model="config.optimizationEnabled"
              type="checkbox"
              aria-label="SIMD Fast Downscale & Crop"
              class="rounded accent-primary w-4 h-4 cursor-pointer"
            />
          </OcrSettingRow>

          <div
            class="grid grid-cols-3 gap-2 pt-2 transition-opacity"
            :class="{ 'opacity-45 pointer-events-none': !config.optimizationEnabled }"
          >
            <div data-setting-id="optimizationTargetMaxSide" class="space-y-1">
              <label class="block text-[11px] text-zinc-600 dark:text-zinc-400">
                Target Max Side <span class="font-mono text-[10px] text-zinc-400">(px)</span>
              </label>
              <input
                v-model.number="config.optimizationTargetMaxSide"
                type="number"
                min="640"
                max="2560"
                step="128"
                :disabled="!config.optimizationEnabled"
                class="w-full px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums focus:outline-none focus:border-primary"
              />
            </div>
            <div data-setting-id="cropPadding" class="col-span-2 grid grid-cols-2 gap-2">
              <div class="space-y-1">
                <label class="block text-[11px] text-zinc-600 dark:text-zinc-400">
                  Crop Pad X <span class="font-mono text-[10px] text-zinc-400">(px)</span>
                </label>
                <input
                  v-model.number="config.cropPaddingX"
                  type="number"
                  min="0"
                  max="50"
                  step="2"
                  :disabled="!config.optimizationEnabled"
                  class="w-full px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums focus:outline-none focus:border-primary"
                />
              </div>
              <div class="space-y-1">
                <label class="block text-[11px] text-zinc-600 dark:text-zinc-400">
                  Crop Pad Y <span class="font-mono text-[10px] text-zinc-400">(ratio)</span>
                </label>
                <input
                  v-model.number="config.cropPaddingY"
                  type="number"
                  min="0"
                  max="0.2"
                  step="0.01"
                  :disabled="!config.optimizationEnabled"
                  class="w-full px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums focus:outline-none focus:border-primary"
                />
              </div>
            </div>
          </div>
          <p v-if="!config.optimizationEnabled" class="mt-1 text-[10px] font-mono text-zinc-400">
            Enable SIMD Fast Downscale &amp; Crop above to send targetMaxSide and crop padding.
          </p>
        </div>
      </div>
    </template>

    <!-- Phase 2: Core Text Detection -->
    <template v-else-if="activeSection === 'core'">
      <div class="space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
            DBNet Input Resolution
          </span>
          <button
            v-if="detectionActiveCount > 0"
            type="button"
            @click="resetGroup('core')"
            class="text-[10px] font-mono text-zinc-400 hover:text-primary cursor-pointer"
          >
            Reset Detection
          </button>
        </div>

        <OcrSettingRow
          setting-id="detectionLimitSideLen"
          label="Detector Input Limit"
          identifier="limitSideLen / limitType"
          description="Constrains input dimensions fed into DBNet detection model."
          tooltip="Higher values detect smaller text at higher memory and inference cost. 'min' scales the shorter side up; 'max' caps the longer side."
          :is-modified="detectionActiveCount > 0"
        >
          <div class="flex items-center gap-1.5">
            <input
              v-model.number="config.detectionLimitSideLen"
              type="number"
              min="320"
              max="3840"
              step="160"
              aria-label="Detector Input Limit Side Length"
              class="w-20 px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-right tabular-nums focus:outline-none focus:border-primary"
            />
            <span class="text-[10px] font-mono text-zinc-400">px</span>
            <select
              v-model="config.detectionLimitType"
              aria-label="Detector Limit Type"
              class="px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 cursor-pointer focus:outline-none focus:border-primary"
            >
              <option value="max">max</option>
              <option value="min">min</option>
            </select>
          </div>
        </OcrSettingRow>

        <div class="flex items-center justify-between gap-2 pt-1">
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">Quick Presets:</span>
          <div class="flex items-center gap-1.5 flex-wrap">
            <button
              v-for="side in sidePresets"
              :key="side"
              type="button"
              @click="config.detectionLimitSideLen = side"
              class="px-2 py-0.5 text-[11px] font-mono rounded border transition-colors cursor-pointer tabular-nums"
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
      </div>
    </template>

    <!-- Phase 3: Post-Detection Filtering -->
    <template v-else>
      <div class="space-y-3">
        <!-- DBNet Contours & Thresholding -->
        <div>
          <div class="flex items-center justify-between mb-1">
            <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
              DBNet Contours &amp; Thresholding
            </span>
            <button
              v-if="postDetectionActiveCount > 0"
              type="button"
              @click="resetGroup('post')"
              class="text-[10px] font-mono text-zinc-400 hover:text-primary cursor-pointer"
            >
              Reset Post-Detection
            </button>
          </div>

          <OcrRangeParam
            setting-id="postprocessThreshold"
            v-model="config.postprocessThreshold"
            label="Pixel Binarization"
            identifier="threshold"
            tooltip="Probability cutoff for classifying a pixel as text vs background during DBNet binarization."
            :min="0.1"
            :max="0.9"
            :step="0.05"
            :decimals="2"
            :default-value="defaultOcrConfig.postprocessThreshold"
          />

          <OcrRangeParam
            setting-id="postprocessBoxThreshold"
            v-model="config.postprocessBoxThreshold"
            label="Box Confidence"
            identifier="boxThreshold"
            tooltip="Minimum average score across a detected polygon required to accept it as a valid text box."
            :min="0.1"
            :max="0.9"
            :step="0.05"
            :decimals="2"
            :default-value="defaultOcrConfig.postprocessBoxThreshold"
          />

          <OcrRangeParam
            setting-id="postprocessUnclipRatio"
            v-model="config.postprocessUnclipRatio"
            label="Polygon Expansion"
            identifier="unclipRatio"
            tooltip="Vatti clipping expansion factor used to restore shrunk DBNet probability regions back to original text boundaries."
            :min="1.0"
            :max="3.0"
            :step="0.1"
            :decimals="1"
            :default-value="defaultOcrConfig.postprocessUnclipRatio"
          />

          <OcrSettingRow
            setting-id="postprocessMaxCandidates"
            label="Max Candidates"
            identifier="maxCandidates"
            description="Maximum number of candidate polygons evaluated per page."
            :is-modified="
              config.postprocessMaxCandidates !== defaultOcrConfig.postprocessMaxCandidates
            "
          >
            <input
              v-model.number="config.postprocessMaxCandidates"
              type="number"
              min="100"
              max="3000"
              step="100"
              aria-label="Max Candidates"
              class="w-20 px-2 py-1 text-xs font-mono text-right rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums focus:outline-none focus:border-primary"
            />
          </OcrSettingRow>

          <OcrSettingRow
            setting-id="postprocessUseDilation"
            label="Morphological Dilation"
            identifier="useDilation"
            description="Applies 2x2 morphological dilation to connect broken character fragments."
            :is-modified="config.postprocessUseDilation !== defaultOcrConfig.postprocessUseDilation"
          >
            <input
              v-model="config.postprocessUseDilation"
              type="checkbox"
              aria-label="Morphological Dilation"
              class="rounded accent-primary w-4 h-4 cursor-pointer"
            />
          </OcrSettingRow>
        </div>

        <!-- Box Geometry Filtering -->
        <div class="pt-2 border-t border-zinc-200/70 dark:border-zinc-800">
          <OcrSettingRow
            setting-id="enableResize"
            label="Custom Bounding Box Geometry Filters"
            identifier="enableResize"
            description="Filter out candidate boxes outside specific pixel dimensions or aspect ratios."
            :is-modified="config.enableResize !== defaultOcrConfig.enableResize"
          >
            <input
              v-model="config.enableResize"
              type="checkbox"
              aria-label="Custom Bounding Box Geometry Filters"
              class="rounded accent-primary w-4 h-4 cursor-pointer"
            />
          </OcrSettingRow>

          <div
            class="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 transition-opacity"
            :class="{ 'opacity-45 pointer-events-none': !config.enableResize }"
          >
            <div class="space-y-1">
              <label class="block text-[10px] text-zinc-500">Max Side (px)</label>
              <input
                v-model.number="config.maxSideLen"
                type="number"
                min="320"
                max="4096"
                step="128"
                :disabled="!config.enableResize"
                class="w-full px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums focus:outline-none focus:border-primary"
              />
            </div>
            <div class="space-y-1">
              <label class="block text-[10px] text-zinc-500">Min Side (px)</label>
              <input
                v-model.number="config.minSideLen"
                type="number"
                min="100"
                max="2048"
                step="64"
                :disabled="!config.enableResize"
                class="w-full px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums focus:outline-none focus:border-primary"
              />
            </div>
            <div class="space-y-1">
              <label class="block text-[10px] text-zinc-500">Min Height (px)</label>
              <input
                v-model.number="config.minHeight"
                type="number"
                min="16"
                max="512"
                step="16"
                :disabled="!config.enableResize"
                class="w-full px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums focus:outline-none focus:border-primary"
              />
            </div>
            <div class="space-y-1">
              <label class="block text-[10px] text-zinc-500">Aspect (-1 auto)</label>
              <input
                v-model.number="config.widthHeightRatio"
                type="number"
                min="-1"
                max="10"
                step="0.1"
                :disabled="!config.enableResize"
                class="w-full px-2 py-1 text-xs font-mono rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 tabular-nums focus:outline-none focus:border-primary"
              />
            </div>
          </div>
          <p v-if="!config.enableResize" class="mt-1 text-[10px] font-mono text-zinc-400">
            Enable Custom Bounding Box Geometry Filters above to override default box bounds.
          </p>
        </div>
      </div>
    </template>
  </OcrSettingsSubTabs>
</template>
