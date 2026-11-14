<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { defaultOcrConfig, type OcrConfig } from '../../../types/ocrConfig';
import {
  availableLayoutModels,
  type LayoutModelOption,
  type OcrModelOption,
} from '../../../types/ocrModels';
import OcrRangeParam from './OcrRangeParam.vue';
import OcrSettingRow from './OcrSettingRow.vue';
import OcrSettingsSubTabs, { type SettingsSubTab } from './OcrSettingsSubTabs.vue';

type RecognitionSection = 'pre' | 'core' | 'post';

const props = defineProps<{
  config: OcrConfig;
  currentModel: OcrModelOption;
  subTab?: RecognitionSection;
}>();

const emit = defineEmits<{
  layoutModelChanged: [];
  'update:subTab': [tab: RecognitionSection];
}>();

const activeSection = ref<RecognitionSection>(props.subTab ?? 'core');

watch(
  () => props.subTab,
  (val) => {
    if (val) activeSection.value = val;
  }
);

watch(activeSection, (val) => {
  emit('update:subTab', val);
});

const selectedLayoutMeta = computed<LayoutModelOption>(
  () =>
    availableLayoutModels.find((m) => m.id === props.config.layoutModel) ??
    availableLayoutModels[0]!
);

const layoutModelBadge = computed<string>(() => {
  if (props.config.layoutModel === 'heuristic' || props.config.layoutModel === 'none') {
    return '0 KB · neural +2.9–6.9 MB';
  }
  return `+${selectedLayoutMeta.value.size} model`;
});

const layoutModelHelp = computed<string>(() => {
  if (props.config.layoutModel === 'heuristic' || props.config.layoutModel === 'none') {
    return 'Built-in mode adds 0 KB. Neural presets add Indra (+2.9 MB), CDLA (+6.9 MB), or PubLayNet (+6.9 MB).';
  }
  return `Downloads extra neural layout weights (+${selectedLayoutMeta.value.size}, ${selectedLayoutMeta.value.assetFile ?? 'neural model'}) on first run.`;
});

const preRecognitionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.layoutModel !== defaultOcrConfig.layoutModel) count++;
  if (props.config.lineClassification !== defaultOcrConfig.lineClassification) count++;
  if (props.config.layoutModel === 'heuristic') {
    if (props.config.lineYThreshold !== defaultOcrConfig.lineYThreshold) count++;
    if (props.config.wordXThreshold !== defaultOcrConfig.wordXThreshold) count++;
  }
  return count;
});

const recognitionActiveCount = computed<number>(() =>
  props.config.recBatchNum === props.currentModel.recommendedBatchSize ? 0 : 1
);

const postRecognitionActiveCount = computed<number>(() =>
  props.config.textScore === defaultOcrConfig.textScore ? 0 : 1
);

function handleLayoutChange() {
  props.config.enableReadingOrder = props.config.layoutModel !== 'none';
  emit('layoutModelChanged');
}

function resetGroup(group: 'pre' | 'core' | 'post') {
  if (group === 'pre') {
    const prevLayout = props.config.layoutModel;
    props.config.layoutModel = defaultOcrConfig.layoutModel;
    props.config.enableReadingOrder = defaultOcrConfig.enableReadingOrder;
    props.config.lineYThreshold = defaultOcrConfig.lineYThreshold;
    props.config.wordXThreshold = defaultOcrConfig.wordXThreshold;
    props.config.lineClassification = defaultOcrConfig.lineClassification;
    if (prevLayout !== props.config.layoutModel) {
      emit('layoutModelChanged');
    }
  } else if (group === 'core') {
    props.config.recBatchNum = props.currentModel.recommendedBatchSize;
  } else {
    props.config.textScore = defaultOcrConfig.textScore;
  }
}

const subTabs = computed<SettingsSubTab[]>(() => [
  {
    id: 'pre',
    label: 'Pre-Recognition',
    badge: '.withPreRecognition()',
    description: 'Document layout, reading order, line grouping & 180° classifier',
    activeCount: preRecognitionActiveCount.value,
  },
  {
    id: 'core',
    label: 'Text Recognition',
    badge: '.withRecognition()',
    description: 'Character recognition batch size and execution tuning',
    activeCount: recognitionActiveCount.value,
  },
  {
    id: 'post',
    label: 'Post-Recognition',
    badge: '.withPostRecognition()',
    description: 'Minimum confidence score cutoff filtering',
    activeCount: postRecognitionActiveCount.value,
  },
]);
</script>

<template>
  <OcrSettingsSubTabs v-model="activeSection" :tabs="subTabs" class="text-xs">
    <!-- Phase 4: Pre-Recognition & Reading Order -->
    <template v-if="activeSection === 'pre'">
      <div class="space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
            Layout &amp; Reading Order
          </span>
          <button
            v-if="preRecognitionActiveCount > 0"
            type="button"
            @click="resetGroup('pre')"
            class="text-[10px] font-mono text-zinc-400 hover:text-primary cursor-pointer"
          >
            Reset Pre-Recognition
          </button>
        </div>

        <div data-setting-id="layoutModel" class="space-y-2">
          <OcrSettingRow
            label="Document Layout &amp; Reading Order"
            identifier=".withLayoutReading()"
            :description="selectedLayoutMeta.description"
            tooltip="Selects geometric heuristic column flow (0 KB), neural layout models (+2.9 MB Indra or +6.9 MB CDLA/PubLayNet), or linear raster scan."
            :model-badge="layoutModelBadge"
            :model-help="layoutModelHelp"
            :is-modified="config.layoutModel !== defaultOcrConfig.layoutModel"
          />

          <select
            v-model="config.layoutModel"
            @change="handleLayoutChange"
            aria-label="Document Layout and Reading Order"
            class="w-full text-xs font-medium rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 py-1.5 px-2.5 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-primary cursor-pointer"
          >
            <optgroup label="Built-in / Zero Download">
              <option value="heuristic">Geometric Column Flow (0 KB)</option>
              <option value="none">Linear Raster Scan (Disabled)</option>
            </optgroup>
            <optgroup label="Neural Layout Models">
              <option value="indra">Indra Neural Reading Order (+2.9 MB)</option>
              <option value="cdla">PP-OCR CDLA Document Layout (+6.9 MB)</option>
              <option value="publaynet">PP-OCR PubLayNet Paper Layout (+6.9 MB)</option>
            </optgroup>
          </select>

          <div
            v-if="config.layoutModel === 'heuristic'"
            data-setting-id="lineThresholds"
            class="space-y-1 pt-1"
          >
            <OcrRangeParam
              v-model="config.lineYThreshold"
              label="Line Y Tolerance"
              identifier="lineYThreshold"
              tooltip="Normalized Y-distance tolerance for grouping text boxes into a common reading line."
              :min="0.1"
              :max="1.5"
              :step="0.05"
              :decimals="2"
              :default-value="defaultOcrConfig.lineYThreshold"
            />
            <OcrRangeParam
              v-model="config.wordXThreshold"
              label="Word X Gap"
              identifier="wordXThreshold"
              tooltip="Horizontal proximity threshold for grouping adjacent characters and words."
              :min="0.1"
              :max="1.5"
              :step="0.05"
              :decimals="2"
              :default-value="defaultOcrConfig.wordXThreshold"
            />
          </div>

          <div
            v-else-if="config.layoutModel !== 'none'"
            class="flex items-center gap-2 p-2 rounded-md bg-primary/5 dark:bg-primary/10 border border-primary/20 text-zinc-600 dark:text-zinc-300 text-[11px]"
          >
            <span class="text-primary font-medium shrink-0">Neural Layout Active:</span>
            <span>Automatic sequence prediction via deep layout geometry.</span>
          </div>

          <div
            v-else
            class="flex items-center gap-2 p-2 rounded-md bg-zinc-50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-zinc-800 text-zinc-500 dark:text-zinc-400 text-[11px]"
          >
            <span class="font-medium shrink-0">Raster Scan:</span>
            <span>Natural top-to-bottom scan without layout or column reordering.</span>
          </div>
        </div>

        <div class="pt-2 border-t border-zinc-200/70 dark:border-zinc-800">
          <OcrSettingRow
            setting-id="lineClassification"
            label="Upside-Down Line Classifier"
            identifier=".withLineClassification()"
            description="Identifies and flips 180° inverted text line crops before recognition."
            tooltip="Runs PP-OCR line orientation classifier on each cropped text line. Enabling downloads an extra cls.mnn neural model (+0.5 MB / 519 KB)."
            model-badge="+0.5 MB model"
            model-help="Adds ~0.5 MB (519 KB) line orientation classifier (cls.mnn) download when enabled."
            :is-modified="config.lineClassification !== defaultOcrConfig.lineClassification"
          >
            <input
              v-model="config.lineClassification"
              type="checkbox"
              aria-label="Upside-Down Line Classifier"
              class="rounded accent-primary w-4 h-4 cursor-pointer"
            />
          </OcrSettingRow>
        </div>
      </div>
    </template>

    <!-- Phase 5: Core Text Recognition -->
    <template v-else-if="activeSection === 'core'">
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
            Batched Sequence Decoding
          </span>
          <span class="text-[10px] font-mono text-zinc-400">
            recommended: {{ currentModel.recommendedBatchSize }}
          </span>
        </div>

        <OcrRangeParam
          setting-id="recBatchNum"
          v-model="config.recBatchNum"
          label="Recognition Batch Size"
          identifier="batchSize"
          :tooltip="
            currentModel.preset === 'indra'
              ? 'Indra Linenet processes lines individually with zero padding overhead (batch size 1 recommended).'
              : 'PP-OCR models execute best with batch size 4–6 to amortize kernel launch latency.'
          "
          :min="1"
          :max="16"
          :step="1"
          :decimals="0"
          :default-value="currentModel.recommendedBatchSize"
        />
      </div>
    </template>

    <!-- Phase 6: Post-Recognition Filtering -->
    <template v-else>
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
            Confidence Filtering
          </span>
          <button
            v-if="postRecognitionActiveCount > 0"
            type="button"
            @click="resetGroup('post')"
            class="text-[10px] font-mono text-zinc-400 hover:text-primary cursor-pointer"
          >
            Reset Post-Recognition
          </button>
        </div>

        <OcrRangeParam
          setting-id="textScore"
          v-model="config.textScore"
          label="Minimum Confidence Cutoff"
          identifier="scoreThreshold"
          tooltip="Filters out low-confidence recognition character sequences below this probability threshold (0.10–0.95)."
          :min="0.1"
          :max="0.95"
          :step="0.05"
          :decimals="2"
          :default-value="defaultOcrConfig.textScore"
        />
      </div>
    </template>
  </OcrSettingsSubTabs>
</template>
