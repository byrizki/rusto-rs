<script setup lang="ts">
import { computed } from 'vue';
import {
  defaultModel,
  modelCategories,
  webDemoModels,
  type ModelCategoryMeta,
  type OcrModelOption,
} from '../../../types/ocrModels';
import type { InferenceBackend } from '../../../utils/ocrReleaseModels';

const props = withDefaults(
  defineProps<{
    selectedModelId: string;
    currentModel: OcrModelOption;
    inferenceBackend?: InferenceBackend;
    autoConfig?: boolean;
  }>(),
  {
    inferenceBackend: undefined,
    autoConfig: false,
  }
);

const emit = defineEmits<{
  select: [modelId: string];
}>();

const groupedModels = computed<Array<{ category: ModelCategoryMeta; models: OcrModelOption[] }>>(
  () =>
    modelCategories
      .map((category) => ({
        category,
        models: webDemoModels.filter((m) => m.category === category.id),
      }))
      .filter((group) => group.models.length > 0)
);

const isDefaultSelected = computed<boolean>(() => props.selectedModelId === defaultModel.id);

function formatCategoryLabel(category: ModelCategoryMeta): string {
  return category.name.replace(/\s*\(Default\)$/u, '');
}

function onChange(event: Event) {
  const target = event.target as HTMLSelectElement;
  emit('select', target.value);
}
</script>

<template>
  <div class="space-y-1.5 text-xs">
    <div class="flex items-center justify-between gap-2">
      <div class="flex items-center gap-1.5">
        <label
          for="ocr-model-selection"
          class="text-[11px] font-semibold text-zinc-700 dark:text-zinc-300"
        >
          Model Selection
        </label>
        <span
          v-if="isDefaultSelected"
          class="px-1.5 py-0.5 text-[10px] font-mono font-medium rounded bg-zinc-200/70 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300"
        >
          default
        </span>
      </div>
      <div class="flex items-center gap-1.5 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>
          preset:
          <strong
            :class="autoConfig ? 'text-primary font-semibold' : 'text-zinc-800 dark:text-zinc-200'"
          >
            {{ autoConfig ? 'auto' : currentModel.preset }}
          </strong>
        </span>
        <span class="text-zinc-300 dark:text-zinc-700">·</span>
        <span class="tabular-nums">batch={{ currentModel.recommendedBatchSize }}</span>
      </div>
    </div>

    <select
      id="ocr-model-selection"
      :value="selectedModelId"
      @change="onChange"
      class="w-full text-xs font-medium rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 py-1.5 px-2.5 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:border-primary cursor-pointer"
    >
      <optgroup
        v-for="group in groupedModels"
        :key="group.category.id"
        :label="formatCategoryLabel(group.category)"
      >
        <option v-for="m in group.models" :key="m.id" :value="m.id">
          {{ m.name }} — {{ m.size }}
        </option>
      </optgroup>
    </select>

    <div class="px-0.5">
      <p class="text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug">
        {{ currentModel.description }}
      </p>
    </div>
  </div>
</template>
