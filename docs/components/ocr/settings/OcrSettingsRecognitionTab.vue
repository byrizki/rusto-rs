<script setup lang="ts">
import { computed, ref } from 'vue';
import type { OcrConfig } from '../../../types/ocrConfig';
import {
  availableLayoutModels,
  type LayoutModelOption,
  type OcrModelOption,
} from '../../../types/ocrModels';
import OcrInfoPopover from './OcrInfoPopover.vue';
import OcrSettingsCollapsibleSection from './OcrSettingsCollapsibleSection.vue';

const props = defineProps<{
  config: OcrConfig;
  currentModel: OcrModelOption;
}>();

const emit = defineEmits<{
  layoutModelChanged: [];
}>();

type RecognitionSection = 'pre' | 'core' | 'post';
const activeSection = ref<RecognitionSection | null>('core');

function onToggle(section: RecognitionSection, nextValue: boolean) {
  if (nextValue) {
    activeSection.value = section;
  } else if (activeSection.value === section) {
    activeSection.value = null;
  }
}

const selectedLayoutMeta = computed<LayoutModelOption>(
  () =>
    availableLayoutModels.find((m) => m.id === props.config.layoutModel) ??
    availableLayoutModels[0]!
);

const preRecognitionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.layoutModel !== 'none' && props.config.layoutModel !== 'heuristic') count++;
  if (props.config.lineClassification) count++;
  if (props.config.layoutModel === 'heuristic') {
    if (props.config.lineYThreshold !== 0.5) count++;
    if (props.config.wordXThreshold !== 0.4) count++;
  }
  return count;
});

const recognitionActiveCount = computed<number>(() => {
  let count = 0;
  const expectedBatch = props.currentModel.preset === 'indra' ? 1 : 4;
  if (props.config.recBatchNum !== expectedBatch) count++;
  return count;
});

const postRecognitionActiveCount = computed<number>(() => {
  let count = 0;
  if (props.config.textScore !== 0.5) count++;
  return count;
});

function handleLayoutChange() {
  props.config.enableReadingOrder = props.config.layoutModel !== 'none';
  emit('layoutModelChanged');
}
</script>

<template>
  <div class="space-y-3 text-xs">
    <!-- Phase 4: Pre-Recognition & Reading Order (Collapsible) -->
    <OcrSettingsCollapsibleSection
      :model-value="activeSection === 'pre'"
      @update:model-value="(val) => onToggle('pre', val)"
      title="Phase 4: Pre-Recognition & Reading Order"
      badge=".withPreRecognition()"
      description="Document layout, reading order, line grouping & 180° classifier"
      :active-count="preRecognitionActiveCount"
    >
      <!-- Layout & Reading Order -->
      <div class="space-y-2.5 pt-1">
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-1.5">
            <label class="font-medium text-zinc-800 dark:text-zinc-200">
              Document Layout & Reading Order
              <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1"
                >.withLayoutReading()</span
              >
            </label>
            <OcrInfoPopover :text="selectedLayoutMeta.description" />
          </div>
          <span class="font-mono text-[10px] text-primary font-semibold tabular-nums">
            {{ selectedLayoutMeta.size }}
          </span>
        </div>

        <select
          v-model="config.layoutModel"
          @change="handleLayoutChange"
          class="w-full text-xs font-medium rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 py-1.5 px-2.5 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-primary cursor-pointer"
        >
          <optgroup label="Built-in / Zero Download">
            <option value="heuristic">Geometric Column Flow (0 KB)</option>
            <option value="none">Linear Raster Scan (Disabled)</option>
          </optgroup>
          <optgroup label="Neural Layout Models">
            <option value="indra">Indra Neural Reading Order (2.9 MB)</option>
            <option value="cdla">PP-OCR CDLA Document Layout (8.4 MB)</option>
            <option value="publaynet">PP-OCR PubLayNet Paper Layout (8.4 MB)</option>
          </optgroup>
        </select>

        <!-- Heuristic Fine-Tuning Controls (Active when heuristic flow is selected) -->
        <div
          v-if="config.layoutModel === 'heuristic'"
          class="grid grid-cols-2 gap-2.5 pt-2 border-t border-zinc-100 dark:border-zinc-800"
        >
          <div class="space-y-1">
            <div class="flex justify-between text-[11px]">
              <div class="flex items-center gap-1">
                <span class="text-zinc-600 dark:text-zinc-400">Line Y Tolerance</span>
                <OcrInfoPopover
                  text="Normalized Y-distance tolerance for grouping text boxes into a common reading line."
                />
              </div>
              <span class="font-mono font-medium tabular-nums">{{
                config.lineYThreshold.toFixed(2)
              }}</span>
            </div>
            <input
              v-model.number="config.lineYThreshold"
              type="range"
              min="0.1"
              max="1.5"
              step="0.05"
              class="w-full accent-primary cursor-pointer"
            />
          </div>
          <div class="space-y-1">
            <div class="flex justify-between text-[11px]">
              <div class="flex items-center gap-1">
                <span class="text-zinc-600 dark:text-zinc-400">Word X Gap</span>
                <OcrInfoPopover
                  text="Horizontal proximity threshold for grouping adjacent characters and words."
                />
              </div>
              <span class="font-mono font-medium tabular-nums">{{
                config.wordXThreshold.toFixed(2)
              }}</span>
            </div>
            <input
              v-model.number="config.wordXThreshold"
              type="range"
              min="0.1"
              max="1.5"
              step="0.05"
              class="w-full accent-primary cursor-pointer"
            />
          </div>
        </div>

        <!-- Neural Layout Active Notice -->
        <div
          v-else-if="config.layoutModel !== 'none'"
          class="flex items-center gap-2 p-2 rounded-md bg-primary/5 dark:bg-primary/10 border border-primary/20 text-zinc-600 dark:text-zinc-300 text-[11px]"
        >
          <span class="text-primary font-medium shrink-0">Neural Layout Active:</span>
          <span>Automatic sequence prediction directly via deep layout geometry.</span>
        </div>

        <!-- Raster Scan Notice -->
        <div
          v-else
          class="flex items-center gap-2 p-2 rounded-md bg-zinc-50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-zinc-800 text-zinc-500 dark:text-zinc-400 text-[11px]"
        >
          <span class="font-medium shrink-0">Raster Scan:</span>
          <span>Natural top-to-bottom scan without layout or column reordering.</span>
        </div>
      </div>

      <!-- Line Orientation Classifier -->
      <div class="pt-2 border-t border-zinc-100 dark:border-zinc-800">
        <label class="flex items-center justify-between gap-3 cursor-pointer">
          <div class="flex items-center gap-1.5 flex-wrap">
            <span class="font-medium text-zinc-800 dark:text-zinc-200">
              Upside-Down Line Classifier
            </span>
            <span class="text-[10px] font-mono text-zinc-400 font-normal"
              >.withLineClassification()</span
            >
            <OcrInfoPopover
              text="Identifies and flips 180° inverted text line crops before character recognition."
            />
          </div>
          <input
            v-model="config.lineClassification"
            type="checkbox"
            class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
          />
        </label>
      </div>
    </OcrSettingsCollapsibleSection>

    <!-- Phase 5: Core Text Recognition (Collapsible) -->
    <OcrSettingsCollapsibleSection
      :model-value="activeSection === 'core'"
      @update:model-value="(val) => onToggle('core', val)"
      title="Phase 5: Text Recognition"
      badge=".withRecognition()"
      description="Character recognition batch size and execution tuning"
      :active-count="recognitionActiveCount"
    >
      <div class="space-y-1.5 pt-1">
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-1.5">
            <label class="font-medium text-zinc-800 dark:text-zinc-200">
              Recognition Batch Size
              <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1">(batchSize)</span>
            </label>
            <OcrInfoPopover
              :text="
                currentModel.preset === 'indra'
                  ? 'Indra Linenet processes lines individually with zero padding overhead (batch size 1 recommended).'
                  : 'PP-OCR models execute best with batch size 4–6 to amortize kernel launch latency.'
              "
            />
          </div>
          <span class="font-mono text-xs text-primary font-semibold tabular-nums">
            {{ config.recBatchNum }}
          </span>
        </div>
        <input
          v-model.number="config.recBatchNum"
          type="range"
          min="1"
          max="16"
          step="1"
          class="w-full accent-primary cursor-pointer"
        />
      </div>
    </OcrSettingsCollapsibleSection>

    <!-- Phase 6: Post-Recognition Filtering (Collapsible) -->
    <OcrSettingsCollapsibleSection
      :model-value="activeSection === 'post'"
      @update:model-value="(val) => onToggle('post', val)"
      title="Phase 6: Post-Recognition Filtering"
      badge=".withPostRecognition()"
      description="Minimum confidence score cutoff filtering"
      :active-count="postRecognitionActiveCount"
    >
      <div class="space-y-2 pt-1">
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-1.5">
            <label class="font-medium text-zinc-800 dark:text-zinc-200">
              Minimum Confidence Cutoff
              <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1"
                >(scoreThreshold)</span
              >
            </label>
            <OcrInfoPopover
              text="Filters out low-confidence recognition character sequences below this probability threshold."
            />
          </div>
          <span class="font-mono text-xs text-primary font-semibold tabular-nums">
            {{ Math.round(config.textScore * 100) }}%
          </span>
        </div>
        <input
          v-model.number="config.textScore"
          type="range"
          min="0.1"
          max="0.95"
          step="0.05"
          class="w-full accent-primary cursor-pointer"
        />
      </div>
    </OcrSettingsCollapsibleSection>
  </div>
</template>
