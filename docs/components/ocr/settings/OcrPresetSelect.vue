<script setup lang="ts">
import type { OcrProfilePreset } from '../../../types/ocrPresets';

defineProps<{
  presets: OcrProfilePreset[];
  activePreset: OcrProfilePreset | null;
  isConfigModified: boolean;
}>();

const emit = defineEmits<{
  select: [preset: OcrProfilePreset];
}>();
</script>

<template>
  <div class="space-y-1.5 text-xs">
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-1.5" role="radiogroup" aria-label="Presets">
      <button
        v-for="preset in presets"
        :key="preset.id"
        type="button"
        role="radio"
        :aria-checked="activePreset?.id === preset.id"
        @click="emit('select', preset)"
        :title="preset.description"
        class="group relative flex items-center justify-between gap-1.5 px-2.5 py-2 rounded-xl border text-left transition-all cursor-pointer active:scale-[0.98]"
        :class="
          activePreset?.id === preset.id
            ? 'border-primary bg-primary/10 text-zinc-900 dark:text-zinc-100 font-semibold shadow-2xs'
            : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800/60'
        "
      >
        <span class="truncate text-[11px]">{{ preset.name }}</span>
        <svg
          v-if="activePreset?.id === preset.id"
          class="w-3.5 h-3.5 text-primary shrink-0"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.5"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </button>
    </div>

    <div
      v-if="activePreset || isConfigModified"
      class="flex items-center gap-1.5 px-0.5 text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug"
    >
      <svg
        v-if="activePreset"
        class="w-3.5 h-3.5 text-primary shrink-0"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
      <span
        v-else
        class="w-2 h-2 rounded-full bg-amber-500 shrink-0"
        title="Custom settings active"
      />
      <p class="truncate">
        {{ activePreset?.description ?? 'Custom pipeline tuning applied.' }}
      </p>
    </div>
  </div>
</template>
