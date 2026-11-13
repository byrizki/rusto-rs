<script setup lang="ts">
import { computed } from 'vue';
import type { OcrDemoStrings } from '../../types/ocrDemoI18n';
import type { OcrModelOption } from '../../types/ocrModels';

const props = defineProps<{
  currentModel: OcrModelOption;
  status: 'idle' | 'initializing' | 'ready' | 'error';
  statusMessage: string;
  isScanning: boolean;
  durationMs: number | null;
  backend?: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu';
  useWebGpu?: boolean;
  isConfigOpen: boolean;
  isCustomConfig: boolean;
  isConfigModified: boolean;
  isExpanded: boolean;
  strings: OcrDemoStrings['toolbar'];
}>();

const activeBackend = computed<'rusto-nn' | 'rusto-rten' | 'rusto-webgpu'>(
  () => props.backend || (props.useWebGpu ? 'rusto-webgpu' : 'rusto-nn')
);

const emit = defineEmits<{
  retryInit: [];
  openRuntimeSettings: [];
  toggleSettings: [];
  toggleExpand: [];
  runOcr: [];
}>();
</script>

<template>
  <header
    class="relative flex flex-wrap items-center justify-between gap-2.5 px-3.5 py-2.5 border-b border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/90 dark:bg-zinc-950/90"
  >
    <div class="flex flex-wrap items-center gap-1.5 min-w-0">
      <div
        class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-zinc-200/80 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-semibold text-zinc-900 dark:text-zinc-100 min-w-0 max-w-[230px]"
        :title="`${currentModel.name} (${currentModel.size})`"
      >
        <svg
          class="w-3.5 h-3.5 text-primary shrink-0"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M12 2 2 7l10 5 10-5-10-5Z" />
          <path d="m2 17 10 5 10-5" />
          <path d="m2 12 10 5 10-5" />
        </svg>
        <span class="truncate">{{ currentModel.name }}</span>
      </div>

      <span
        v-if="status === 'ready'"
        class="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 border border-emerald-200/70 dark:border-emerald-900/60"
        :title="statusMessage || strings.statusReady"
        :aria-label="strings.statusReady"
      >
        <svg
          class="w-3.5 h-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M20 6 9 17l-5-5" />
        </svg>
      </span>

      <span
        v-else-if="status === 'initializing'"
        class="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-amber-50 dark:bg-amber-950/50 text-amber-600 dark:text-amber-400 border border-amber-200/70 dark:border-amber-900/60"
        :title="statusMessage || strings.statusLoading"
        :aria-label="strings.statusLoading"
      >
        <svg
          class="w-3.5 h-3.5 animate-spin"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          aria-hidden="true"
        >
          <path d="M21 12a9 9 0 1 1-6.219-8.56" />
        </svg>
      </span>

      <button
        v-else-if="status === 'error'"
        type="button"
        @click="emit('retryInit')"
        class="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-900 hover:bg-red-100 dark:hover:bg-red-950 transition-colors cursor-pointer"
        :title="`${statusMessage || strings.statusError} — ${strings.retry}`"
        :aria-label="strings.retry"
      >
        <svg
          class="w-3.5 h-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
          <path d="M3 3v5h5" />
          <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
        </svg>
      </button>

      <button
        type="button"
        @click="emit('openRuntimeSettings')"
        class="hidden sm:inline-flex items-center justify-center w-7 h-7 rounded-lg border transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        :class="
          activeBackend === 'rusto-webgpu'
            ? 'bg-purple-50 dark:bg-purple-950/40 border-purple-200 dark:border-purple-900/70 text-purple-700 dark:text-purple-300 hover:bg-purple-100/70 dark:hover:bg-purple-950/70'
            : activeBackend === 'rusto-rten'
              ? 'bg-blue-50 dark:bg-blue-950/40 border-blue-200 dark:border-blue-900/70 text-blue-700 dark:text-blue-300 hover:bg-blue-100/70 dark:hover:bg-blue-950/70'
              : 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-900/70 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100/70 dark:hover:bg-emerald-950/70'
        "
        :title="`Runtime: ${activeBackend === 'rusto-webgpu' ? 'WebGPU' : activeBackend}`"
        :aria-label="`Runtime: ${activeBackend}`"
      >
        <svg
          v-if="activeBackend === 'rusto-webgpu'"
          class="w-3.5 h-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
        </svg>
        <svg
          v-else
          class="w-3.5 h-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <rect x="4" y="4" width="16" height="16" rx="2" />
          <rect x="9" y="9" width="6" height="6" />
          <path d="M15 2v2" />
          <path d="M15 20v2" />
          <path d="M2 15h2" />
          <path d="M2 9h2" />
          <path d="M20 15h2" />
          <path d="M20 9h2" />
          <path d="M9 2v2" />
          <path d="M9 20v2" />
        </svg>
      </button>
    </div>

    <div class="flex items-center gap-1.5 sm:gap-2 ml-auto">
      <span
        v-if="durationMs !== null"
        class="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-mono font-medium text-zinc-600 dark:text-zinc-300 bg-zinc-100 dark:bg-zinc-900 tabular-nums"
      >
        <svg
          class="w-3 h-3 text-primary shrink-0"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            d="M11.983 1.907a.75.75 0 0 0-1.292-.657l-8.5 9.5A.75.75 0 0 0 2.75 12h6.572l-1.305 6.093a.75.75 0 0 0 1.292.657l8.5-9.5A.75.75 0 0 0 17.25 8h-6.572l1.305-6.093Z"
          />
        </svg>
        {{ durationMs }} ms
      </span>

      <button
        type="button"
        @click="emit('toggleSettings')"
        class="relative inline-flex items-center justify-center p-1.5 rounded-lg border transition-all cursor-pointer active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        :class="
          isConfigOpen
            ? 'border-primary bg-primary/10 text-primary'
            : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800'
        "
        :title="strings.settings"
        :aria-label="strings.settings"
      >
        <svg
          class="w-4 h-4 shrink-0"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <line x1="4" x2="4" y1="21" y2="14" />
          <line x1="4" x2="4" y1="10" y2="3" />
          <line x1="12" x2="12" y1="21" y2="12" />
          <line x1="12" x2="12" y1="8" y2="3" />
          <line x1="20" x2="20" y1="21" y2="16" />
          <line x1="20" x2="20" y1="12" y2="3" />
          <line x1="2" x2="6" y1="14" y2="14" />
          <line x1="10" x2="14" y1="8" y2="8" />
          <line x1="18" x2="22" y1="16" y2="16" />
        </svg>
        <span
          v-if="isCustomConfig || isConfigModified"
          class="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-primary border border-white dark:border-zinc-900"
        />
      </button>

      <button
        type="button"
        @click="emit('toggleExpand')"
        class="hidden sm:inline-flex items-center justify-center p-1.5 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-all cursor-pointer active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        :title="isExpanded ? strings.exitExpandedView : strings.expandView"
        :aria-label="isExpanded ? strings.exitExpandedView : strings.expandView"
      >
        <svg
          v-if="!isExpanded"
          class="w-3.5 h-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <polyline points="15 3 21 3 21 9" />
          <polyline points="9 21 3 21 3 15" />
          <line x1="21" y1="3" x2="14" y2="10" />
          <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
        <svg
          v-else
          class="w-3.5 h-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <polyline points="4 14 10 14 10 20" />
          <polyline points="20 10 14 10 14 4" />
          <line x1="14" y1="10" x2="21" y2="3" />
          <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
      </button>

      <button
        type="button"
        :disabled="status === 'initializing' || isScanning"
        @click="emit('runOcr')"
        class="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-primary text-white hover:opacity-95 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all cursor-pointer shadow-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
      >
        <span
          v-if="isScanning || status === 'initializing'"
          class="inline-block w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"
        />
        <svg
          v-else
          class="w-3.5 h-3.5 shrink-0"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            d="M6.3 2.84A1.5 1.5 0 0 0 4 4.11v11.78a1.5 1.5 0 0 0 2.3 1.27l9.344-5.891a1.5 1.5 0 0 0 0-2.538L6.3 2.841Z"
          />
        </svg>
        <span>
          {{
            isScanning
              ? strings.processing
              : status === 'initializing'
                ? strings.downloading
                : isConfigModified
                  ? strings.rerunOcr
                  : strings.runOcr
          }}
        </span>
      </button>
    </div>
  </header>
</template>
