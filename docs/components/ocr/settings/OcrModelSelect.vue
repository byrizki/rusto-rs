<script setup lang="ts">
import { computed } from 'vue';
import {
  defaultModel,
  modelCategories,
  webDemoModels,
  type ModelCategoryMeta,
  type OcrModelOption,
} from '../../../types/ocrModels';
import { isModelSupportedByBackend, type InferenceBackend } from '../../../utils/ocrReleaseModels';

const props = defineProps<{
  selectedModelId: string;
  currentModel: OcrModelOption;
  inferenceBackend: InferenceBackend;
}>();

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

const backendBadges = computed<
  Array<{ id: InferenceBackend; label: string; supported: boolean; active: boolean }>
>(() => [
  {
    id: 'rusto-nn',
    label: 'rusto-nn',
    supported: isModelSupportedByBackend(props.currentModel.id, 'rusto-nn'),
    active: props.inferenceBackend === 'rusto-nn',
  },
  {
    id: 'rusto-rten',
    label: 'rusto-rten',
    supported: isModelSupportedByBackend(props.currentModel.id, 'rusto-rten'),
    active: props.inferenceBackend === 'rusto-rten',
  },
  {
    id: 'rusto-webgpu',
    label: 'rusto-webgpu',
    supported: isModelSupportedByBackend(props.currentModel.id, 'rusto-webgpu'),
    active: props.inferenceBackend === 'rusto-webgpu',
  },
]);

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
          <strong class="text-zinc-800 dark:text-zinc-200">{{ currentModel.preset }}</strong>
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

    <div class="flex items-center justify-between gap-2 px-0.5">
      <p class="text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug truncate">
        {{ currentModel.description }}
      </p>
      <div class="flex items-center gap-1 shrink-0 font-mono text-[10px]">
        <span
          v-for="b in backendBadges"
          :key="b.id"
          class="px-1.5 py-0.5 rounded border"
          :class="
            b.supported
              ? b.active
                ? 'border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 font-semibold'
                : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-500 dark:text-zinc-400'
              : 'border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-zinc-900 text-zinc-400 line-through'
          "
          :title="
            b.supported
              ? b.active
                ? `${b.label}: active runtime`
                : `${b.label}: supported`
              : `${b.label}: unsupported (RTen supports FP32 models only; FP16/INT8 require rusto-nn or rusto-webgpu)`
          "
        >
          {{ b.label }}
        </span>
      </div>
    </div>
  </div>
</template>
