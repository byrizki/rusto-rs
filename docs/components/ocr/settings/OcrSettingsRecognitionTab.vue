<script setup lang="ts">
import { computed } from 'vue';
import type { OcrConfig } from '../../../types/ocrConfig';
import {
  availableLayoutModels,
  type LayoutModelOption,
  type OcrModelOption,
} from '../../../types/ocrModels';

const props = defineProps<{
  config: OcrConfig;
  currentModel: OcrModelOption;
}>();

const emit = defineEmits<{
  layoutModelChanged: [];
}>();

const selectedLayoutMeta = computed<LayoutModelOption>(
  () =>
    availableLayoutModels.find((m) => m.id === props.config.layoutModel) ??
    availableLayoutModels[0]!
);

function handleLayoutChange() {
  props.config.enableReadingOrder = props.config.layoutModel !== 'none';
  emit('layoutModelChanged');
}
</script>

<template>
  <div class="space-y-4 text-xs">
    <!-- Phase 4: Pre-Recognition -->
    <div
      class="flex items-center justify-between pb-1 border-b border-zinc-100 dark:border-zinc-800"
    >
      <span class="text-[11px] font-semibold text-zinc-900 dark:text-zinc-100">
        Phase 4: Pre-Recognition & Grouping
      </span>
      <span
        class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium"
      >
        .withPreRecognition()
      </span>
    </div>

    <!-- Layout & Reading Order -->
    <div
      class="space-y-2.5 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <div class="flex justify-between items-center">
        <label class="font-medium text-zinc-800 dark:text-zinc-200">
          Document Layout & Reading Order
          <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1"
            >.withLayoutReading()</span
          >
        </label>
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

      <p class="text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug">
        {{ selectedLayoutMeta.description }}
      </p>

      <div class="grid grid-cols-2 gap-2.5 pt-2 border-t border-zinc-200/60 dark:border-zinc-800">
        <div class="space-y-1">
          <div class="flex justify-between text-[11px]">
            <span class="text-zinc-600 dark:text-zinc-400">Line Y Tolerance</span>
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
            <span class="text-zinc-600 dark:text-zinc-400">Word X Gap</span>
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
    </div>

    <!-- Line Orientation Classifier -->
    <div
      class="p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <label class="flex items-center justify-between gap-3 cursor-pointer">
        <div>
          <span class="font-medium text-zinc-800 dark:text-zinc-200 block">
            Upside-Down Line Classifier
            <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1"
              >.withLineClassification()</span
            >
          </span>
          <span class="text-[11px] text-zinc-500 dark:text-zinc-400">
            Identifies and flips 180° inverted text line crops before character recognition
          </span>
        </div>
        <input
          v-model="config.lineClassification"
          type="checkbox"
          class="rounded accent-primary w-4 h-4 cursor-pointer shrink-0"
        />
      </label>
    </div>

    <!-- Phase 5: Recognition -->
    <div
      class="flex items-center justify-between pt-2 pb-1 border-b border-zinc-100 dark:border-zinc-800"
    >
      <span class="text-[11px] font-semibold text-zinc-900 dark:text-zinc-100">
        Phase 5: Text Recognition
      </span>
      <span
        class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium"
      >
        .withRecognition()
      </span>
    </div>

    <div
      class="space-y-2 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <div class="flex justify-between items-center">
        <label class="font-medium text-zinc-800 dark:text-zinc-200">
          Recognition Batch Size
          <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1">(batchSize)</span>
        </label>
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
      <p class="text-[11px] text-zinc-500 dark:text-zinc-400">
        {{
          currentModel.preset === 'indra'
            ? 'Indra Linenet processes lines individually with zero padding overhead (batch size 1 recommended).'
            : 'PP-OCR models execute best with batch size 4–6 to amortize kernel launch latency.'
        }}
      </p>
    </div>

    <!-- Phase 6: Post-Recognition -->
    <div
      class="flex items-center justify-between pt-2 pb-1 border-b border-zinc-100 dark:border-zinc-800"
    >
      <span class="text-[11px] font-semibold text-zinc-900 dark:text-zinc-100">
        Phase 6: Post-Recognition Filtering
      </span>
      <span
        class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium"
      >
        .withPostRecognition()
      </span>
    </div>

    <div
      class="space-y-2 p-3 rounded-xl border border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50"
    >
      <div class="flex justify-between items-center">
        <label class="font-medium text-zinc-800 dark:text-zinc-200">
          Minimum Confidence Cutoff
          <span class="text-[10px] font-mono text-zinc-400 font-normal ml-1">(scoreThreshold)</span>
        </label>
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
      <p class="text-[11px] text-zinc-500 dark:text-zinc-400">
        Filters out low-confidence recognition character sequences below this probability threshold.
      </p>
    </div>
  </div>
</template>
