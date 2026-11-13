<script setup lang="ts">
import { computed } from 'vue';
import type { OcrSettingsTab } from '../../composables/useWebOcrSettings';
import type { OcrConfig } from '../../types/ocrConfig';
import type { OcrDemoStrings } from '../../types/ocrDemoI18n';
import type { OcrModelOption } from '../../types/ocrModels';
import { ocrProfilePresets, type OcrProfilePreset } from '../../types/ocrPresets';
import OcrPresetSelect from './settings/OcrPresetSelect.vue';
import OcrSettingsDetectionTab from './settings/OcrSettingsDetectionTab.vue';
import OcrSettingsRecognitionTab from './settings/OcrSettingsRecognitionTab.vue';
import OcrSettingsRuntimeTab from './settings/OcrSettingsRuntimeTab.vue';

const props = defineProps<{
  isOpen: boolean;
  activeTab: OcrSettingsTab;
  config: OcrConfig;
  currentModel: OcrModelOption;
  selectedModelId: string;
  baseURL: string;
  isWebGpuSupported: boolean;
  isConfigModified: boolean;
  hasRun: boolean;
  isPresetActive: (preset: OcrProfilePreset, selectedModelId?: string) => boolean;
  strings: OcrDemoStrings['settings'];
}>();

const emit = defineEmits<{
  'update:activeTab': [tab: OcrSettingsTab];
  close: [];
  resetDefaults: [];
  applyPreset: [preset: OcrProfilePreset];
  layoutModelChanged: [];
  backendChanged: [backend: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu'];
  applyAndRun: [];
}>();

const activePreset = computed<OcrProfilePreset | null>(() => {
  return ocrProfilePresets.find((p) => props.isPresetActive(p, props.selectedModelId)) ?? null;
});

const tabs = computed<Array<{ id: OcrSettingsTab; label: string }>>(() => [
  { id: 'detection', label: props.strings.tabDetection },
  { id: 'recognition', label: props.strings.tabRecognition },
  { id: 'runtime', label: props.strings.tabRuntime },
]);
</script>

<template>
  <Transition
    enter-active-class="transition duration-150 ease-out"
    enter-from-class="opacity-0 scale-98"
    enter-to-class="opacity-100 scale-100"
    leave-active-class="transition duration-100 ease-in"
    leave-from-class="opacity-100 scale-100"
    leave-to-class="opacity-0 scale-98"
  >
    <div
      v-if="isOpen"
      class="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/40 dark:bg-black/60 backdrop-blur-xs"
      @click.self="emit('close')"
    >
      <div
        role="dialog"
        aria-modal="true"
        :aria-label="strings.title"
        class="w-full max-w-[560px] max-h-[85vh] sm:max-h-[660px] flex flex-col rounded-2xl shadow-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs overflow-hidden"
      >
        <div
          class="flex items-center justify-between px-4 py-3 border-b border-zinc-200/80 dark:border-zinc-800 shrink-0"
        >
          <div class="flex items-center gap-2">
            <svg
              class="w-4 h-4 text-primary shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <line x1="4" y1="21" x2="4" y2="14" />
              <line x1="4" y1="10" x2="4" y2="3" />
              <line x1="12" y1="21" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12" y2="3" />
              <line x1="20" y1="21" x2="20" y2="16" />
              <line x1="20" y1="12" x2="20" y2="3" />
              <line x1="1" y1="14" x2="7" y2="14" />
              <line x1="9" y1="8" x2="15" y2="8" />
              <line x1="17" y1="16" x2="23" y2="16" />
            </svg>
            <h3 class="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              {{ strings.title }}
            </h3>
          </div>
          <div class="flex items-center gap-1.5">
            <button
              type="button"
              @click="emit('resetDefaults')"
              :title="strings.resetDefaults"
              :aria-label="strings.resetDefaults"
              class="p-1.5 rounded-lg text-zinc-400 hover:text-primary hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors cursor-pointer"
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
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
            </button>
            <button
              type="button"
              @click="emit('close')"
              class="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors cursor-pointer"
              aria-label="Close settings"
            >
              <svg
                class="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
              >
                <path d="M18 6 6 18" />
                <path d="m6 6 12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div
          class="px-4 py-2.5 border-b border-zinc-100 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50 shrink-0"
        >
          <OcrPresetSelect
            :presets="ocrProfilePresets"
            :active-preset="activePreset"
            :is-config-modified="isConfigModified"
            @select="(preset) => emit('applyPreset', preset)"
          />
        </div>

        <div class="px-4 pt-2.5 pb-2 border-b border-zinc-200/70 dark:border-zinc-800 shrink-0">
          <div
            class="grid grid-cols-3 gap-1 p-1 rounded-xl bg-zinc-100/80 dark:bg-zinc-950/80 border border-zinc-200/60 dark:border-zinc-800"
            role="tablist"
          >
            <button
              v-for="tab in tabs"
              :key="tab.id"
              type="button"
              role="tab"
              :aria-selected="activeTab === tab.id"
              @click="emit('update:activeTab', tab.id)"
              class="inline-flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg text-xs font-medium transition-all cursor-pointer"
              :class="
                activeTab === tab.id
                  ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 font-semibold shadow-2xs'
                  : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
              "
            >
              <svg
                v-if="tab.id === 'detection'"
                class="w-3.5 h-3.5 shrink-0"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
              >
                <path d="M3 7V5a2 2 0 0 1 2-2h2" />
                <path d="M17 3h2a2 2 0 0 1 2 2v2" />
                <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
                <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
                <rect width="10" height="8" x="7" y="8" rx="1" />
              </svg>
              <svg
                v-else-if="tab.id === 'recognition'"
                class="w-3.5 h-3.5 shrink-0"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
              >
                <polyline points="4 7 4 4 20 4 20 7" />
                <line x1="9" y1="20" x2="15" y2="20" />
                <line x1="12" y1="4" x2="12" y2="20" />
              </svg>
              <svg
                v-else
                class="w-3.5 h-3.5 shrink-0"
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
              <span>{{ tab.label }}</span>
            </button>
          </div>
        </div>

        <div class="flex-1 min-h-0 p-4 overflow-y-auto">
          <OcrSettingsDetectionTab v-if="activeTab === 'detection'" :config="config" />
          <OcrSettingsRecognitionTab
            v-else-if="activeTab === 'recognition'"
            :config="config"
            :current-model="currentModel"
            @layout-model-changed="emit('layoutModelChanged')"
          />
          <OcrSettingsRuntimeTab
            v-else
            :config="config"
            :current-model="currentModel"
            :selected-model-id="selectedModelId"
            :base-u-r-l="baseURL"
            :is-web-gpu-supported="isWebGpuSupported"
            @backend-changed="(b) => emit('backendChanged', b)"
          />
        </div>

        <div
          class="px-4 py-2.5 border-t border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/80 dark:bg-zinc-950/60 flex items-center justify-between gap-2 shrink-0"
        >
          <span
            v-if="isConfigModified"
            class="inline-flex items-center justify-center w-6 h-6 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400"
            :title="strings.modifiedHint"
            :aria-label="strings.modifiedHint"
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
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </span>
          <span v-else />

          <div class="flex items-center gap-2 ml-auto">
            <button
              type="button"
              @click="emit('close')"
              class="px-3 py-1.5 rounded-lg text-xs font-medium border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors cursor-pointer"
            >
              {{ strings.close }}
            </button>
            <button
              v-if="isConfigModified || !hasRun"
              type="button"
              @click="emit('applyAndRun')"
              class="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-primary text-white hover:opacity-95 active:scale-[0.98] transition-all cursor-pointer shadow-xs"
            >
              {{ strings.applyAndRun }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>
