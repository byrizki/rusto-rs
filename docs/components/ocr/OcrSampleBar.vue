<script setup lang="ts">
import type { OcrDemoStrings } from '../../types/ocrDemoI18n';
import type { SampleImage } from '../../types/ocrSamples';

defineProps<{
  samples: SampleImage[];
  selectedSample: string;
  customFileName: string | null;
  disabled: boolean;
  strings: OcrDemoStrings['samples'];
}>();

const emit = defineEmits<{
  selectSample: [sample: SampleImage];
  uploadFile: [file: File];
}>();

function onInputChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) {
    emit('uploadFile', file);
    input.value = '';
  }
}
</script>

<template>
  <div
    class="px-3.5 py-2 border-b border-zinc-200/80 dark:border-zinc-800 bg-white dark:bg-zinc-900 flex flex-wrap items-center justify-between gap-2"
  >
    <div class="flex items-center gap-1.5 overflow-x-auto max-w-full py-0.5">
      <span
        class="text-[11px] font-medium text-zinc-500 dark:text-zinc-400 shrink-0 mr-1 hidden sm:inline"
      >
        {{ strings.label }}:
      </span>

      <button
        v-for="sample in samples"
        :key="sample.id"
        type="button"
        :disabled="disabled"
        @click="emit('selectSample', sample)"
        :title="sample.description"
        class="group inline-flex items-center gap-1.5 pl-1 pr-2.5 py-1 text-xs rounded-lg border transition-all cursor-pointer shrink-0 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        :class="
          selectedSample === sample.id
            ? 'border-primary bg-primary/10 text-primary font-semibold'
            : 'border-zinc-200 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50 text-zinc-700 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-zinc-700 hover:bg-zinc-100/70 dark:hover:bg-zinc-800/60'
        "
      >
        <img
          :src="sample.url"
          :alt="strings.names[sample.id] || sample.name"
          loading="lazy"
          class="w-5 h-5 rounded object-cover border border-black/10 dark:border-white/10 shrink-0 bg-zinc-200 dark:bg-zinc-800"
        />
        <span>{{ strings.names[sample.id] || sample.name }}</span>
      </button>

      <span
        v-if="selectedSample === 'custom' && customFileName"
        class="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg border border-primary bg-primary/10 text-primary font-semibold shrink-0 max-w-[180px]"
        :title="customFileName"
      >
        <svg
          class="w-3.5 h-3.5 shrink-0"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fill-rule="evenodd"
            d="M1 5.25A2.25 2.25 0 0 1 3.25 3h13.5A2.25 2.25 0 0 1 19 5.25v9.5A2.25 2.25 0 0 1 16.75 17H3.25A2.25 2.25 0 0 1 1 14.75v-9.5Zm1.5 5.81v3.69c0 .414.336.75.75.75h13.5a.75.75 0 0 0 .75-.75v-2.69l-2.22-2.219a.75.75 0 0 0-1.06 0l-1.91 1.909.47.47a.75.75 0 1 1-1.06 1.06L6.53 8.091a.75.75 0 0 0-1.06 0l-2.97 2.97ZM12 7a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
            clip-rule="evenodd"
          />
        </svg>
        <span class="truncate">{{ customFileName }}</span>
      </span>
    </div>

    <div class="flex items-center gap-2 shrink-0 ml-auto">
      <span class="text-[11px] text-zinc-400 dark:text-zinc-500 hidden xl:inline">
        {{ strings.dropOrPasteHint }}
      </span>

      <label
        class="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-lg border border-dashed border-zinc-300 dark:border-zinc-700 bg-zinc-50/60 dark:bg-zinc-950/60 text-zinc-700 dark:text-zinc-300 hover:border-primary hover:text-primary hover:bg-primary/5 transition-all cursor-pointer active:scale-[0.98]"
        :class="{ 'opacity-50 pointer-events-none': disabled }"
      >
        <svg
          class="w-3.5 h-3.5 shrink-0"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            d="M9.25 13.25a.75.75 0 0 0 1.5 0V4.636l2.955 3.129a.75.75 0 0 0 1.09-1.03l-4.25-4.5a.75.75 0 0 0-1.09 0l-4.25 4.5a.75.75 0 1 0 1.09 1.03L9.25 4.636v8.614Z"
          />
          <path
            d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z"
          />
        </svg>
        <span>{{ strings.uploadBtn }}</span>
        <input
          type="file"
          accept="image/*"
          class="hidden"
          :disabled="disabled"
          @change="onInputChange"
        />
      </label>
    </div>
  </div>
</template>
