<script setup lang="ts">
import { computed } from 'vue';
import type { OcrDemoStrings } from '../../types/ocrDemoI18n';
import { availableModels, modelCategories, type OcrModelOption } from '../../types/ocrModels';

const props = defineProps<{
  selectedModel: string;
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
  'update:selectedModel': [modelId: string];
  retryInit: [];
  openRuntimeSettings: [];
  toggleSettings: [];
  toggleExpand: [];
  runOcr: [];
}>();

function onModelChange(event: Event) {
  const target = event.target as HTMLSelectElement;
  emit('update:selectedModel', target.value);
}
</script>

<template>
  <header
    class="relative flex flex-wrap items-center justify-between gap-2.5 px-3.5 py-2.5 border-b border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/90 dark:bg-zinc-950/90"
  >
    <div class="flex flex-wrap items-center gap-2 min-w-0">
      <div class="inline-flex items-center gap-2 min-w-0">
        <label
          for="ocr-model-select"
          class="text-xs font-medium text-zinc-500 dark:text-zinc-400 hidden sm:inline"
        >
          {{ strings.modelLabel }}
        </label>

        <div class="relative inline-flex items-center">
          <select
            id="ocr-model-select"
            :value="selectedModel"
            :disabled="status === 'initializing' || isScanning"
            @change="onModelChange"
            class="appearance-none pl-2.5 pr-7 py-1.5 text-xs font-semibold rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 hover:border-zinc-300 dark:hover:border-zinc-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:border-primary disabled:opacity-50 disabled:cursor-not-allowed transition-colors enabled:cursor-pointer max-w-[220px] sm:max-w-[250px] truncate"
          >
            <optgroup v-for="cat in modelCategories" :key="cat.id" :label="cat.name">
              <option
                v-for="model in availableModels.filter((m) => m.category === cat.id)"
                :key="model.id"
                :value="model.id"
              >
                {{ model.name }} ({{ model.size }})
              </option>
            </optgroup>
          </select>
          <span
            class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2 text-zinc-400"
          >
            <svg class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path
                fill-rule="evenodd"
                d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                clip-rule="evenodd"
              />
            </svg>
          </span>
        </div>
      </div>

      <div class="inline-flex items-center gap-1.5" :title="statusMessage">
        <span
          v-if="status === 'ready'"
          class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200/70 dark:border-emerald-900/60"
        >
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          {{ strings.statusReady }}
        </span>

        <span
          v-else-if="status === 'initializing'"
          class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border border-amber-200/70 dark:border-amber-900/60"
        >
          <span class="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
          {{ strings.statusLoading }}
        </span>

        <button
          v-else-if="status === 'error'"
          type="button"
          @click="emit('retryInit')"
          class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium bg-red-50 dark:bg-red-950/50 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-900 hover:bg-red-100 dark:hover:bg-red-950 transition-colors cursor-pointer"
        >
          <span class="w-1.5 h-1.5 rounded-full bg-red-500" />
          {{ strings.retry }}
        </button>

        <span
          v-else
          class="hidden md:inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-mono text-zinc-500 dark:text-zinc-400 bg-zinc-100 dark:bg-zinc-900 border border-zinc-200/70 dark:border-zinc-800 tabular-nums"
        >
          {{ currentModel.size }}
        </span>
      </div>

      <button
        type="button"
        @click="emit('openRuntimeSettings')"
        class="hidden sm:inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium border transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        :class="
          activeBackend === 'rusto-webgpu'
            ? 'bg-purple-50 dark:bg-purple-950/40 border-purple-200 dark:border-purple-900/70 text-purple-700 dark:text-purple-300 hover:bg-purple-100/70 dark:hover:bg-purple-950/70'
            : activeBackend === 'rusto-rten'
              ? 'bg-blue-50 dark:bg-blue-950/40 border-blue-200 dark:border-blue-900/70 text-blue-700 dark:text-blue-300 hover:bg-blue-100/70 dark:hover:bg-blue-950/70'
              : 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-900/70 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100/70 dark:hover:bg-emerald-950/70'
        "
        :title="`Inference Engine: ${activeBackend}. Click to switch.`"
      >
        <span
          class="w-1.5 h-1.5 rounded-full"
          :class="
            activeBackend === 'rusto-webgpu'
              ? 'bg-purple-500'
              : activeBackend === 'rusto-rten'
                ? 'bg-blue-500'
                : 'bg-emerald-500'
          "
        />
        <span>{{ activeBackend === 'rusto-webgpu' ? 'WebGPU' : activeBackend }}</span>
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
        class="relative inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border transition-all cursor-pointer active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
        :class="
          isConfigOpen
            ? 'border-primary bg-primary/10 text-primary'
            : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800'
        "
        :title="strings.settings"
      >
        <svg
          class="w-3.5 h-3.5 shrink-0"
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
        <span class="hidden sm:inline">{{ strings.settings }}</span>
        <span
          v-if="isCustomConfig || isConfigModified"
          class="w-1.5 h-1.5 rounded-full bg-primary shrink-0"
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
