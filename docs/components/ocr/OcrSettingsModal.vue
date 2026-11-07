<script setup lang="ts">
import { computed } from 'vue';
import type { OcrSettingsTab } from '../../composables/useWebOcrSettings';
import type { OcrConfig } from '../../types/ocrConfig';
import type { OcrDemoStrings } from '../../types/ocrDemoI18n';
import type { OcrModelOption } from '../../types/ocrModels';
import { ocrProfilePresets, type OcrProfilePreset } from '../../types/ocrPresets';
import OcrSettingsDetectionTab from './settings/OcrSettingsDetectionTab.vue';
import OcrSettingsPreDetectionTab from './settings/OcrSettingsPreDetectionTab.vue';
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
  isPresetActive: (preset: OcrProfilePreset) => boolean;
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

const activePresetDescription = computed(() => {
  const match = ocrProfilePresets.find((p) => props.isPresetActive(p));
  return match?.description ?? null;
});

const tabs = computed<Array<{ id: OcrSettingsTab; label: string }>>(() => [
  { id: 'preDetection', label: props.strings.tabPreDetection },
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
      class="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/35 dark:bg-black/55 backdrop-blur-xs"
      @click.self="emit('close')"
    >
      <div
        role="dialog"
        aria-modal="true"
        :aria-label="strings.title"
        class="w-full max-w-[520px] max-h-[85dvh] flex flex-col rounded-2xl shadow-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs overflow-hidden"
      >
        <div
          class="flex items-center justify-between px-4 py-3 border-b border-zinc-200/80 dark:border-zinc-800 shrink-0"
        >
          <h3 class="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            {{ strings.title }}
          </h3>
          <div class="flex items-center gap-3">
            <button
              type="button"
              @click="emit('resetDefaults')"
              class="text-xs text-zinc-500 hover:text-primary transition-colors cursor-pointer font-medium"
            >
              {{ strings.resetDefaults }}
            </button>
            <button
              type="button"
              @click="emit('close')"
              class="p-1 rounded-lg text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 cursor-pointer"
              aria-label="Close settings"
            >
              ✕
            </button>
          </div>
        </div>

        <div
          class="px-4 py-2.5 border-b border-zinc-100 dark:border-zinc-800 bg-zinc-50/80 dark:bg-zinc-950/60 shrink-0 space-y-1.5"
        >
          <div class="flex items-center gap-1.5 flex-wrap">
            <span
              class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mr-0.5 shrink-0"
            >
              {{ strings.presetsLabel }}:
            </span>
            <button
              v-for="preset in ocrProfilePresets"
              :key="preset.id"
              type="button"
              @click="emit('applyPreset', preset)"
              :title="preset.description"
              class="px-2 py-0.5 text-[11px] rounded-md border transition-all cursor-pointer"
              :class="
                isPresetActive(preset)
                  ? 'bg-primary text-white border-primary font-semibold shadow-2xs'
                  : 'border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-300 hover:border-zinc-400 dark:hover:border-zinc-600'
              "
            >
              {{ preset.name }}
            </button>
          </div>
          <p
            v-if="activePresetDescription"
            class="text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug"
          >
            {{ activePresetDescription }}
          </p>
        </div>

        <div
          class="grid grid-cols-4 gap-1 px-4 pt-2.5 pb-2 border-b border-zinc-200/70 dark:border-zinc-800 shrink-0 bg-white dark:bg-zinc-900"
          role="tablist"
        >
          <button
            v-for="tab in tabs"
            :key="tab.id"
            type="button"
            role="tab"
            :aria-selected="activeTab === tab.id"
            @click="emit('update:activeTab', tab.id)"
            class="py-1.5 px-2 rounded-lg text-xs font-medium transition-colors cursor-pointer text-center"
            :class="
              activeTab === tab.id
                ? 'bg-primary/10 text-primary font-semibold'
                : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200 hover:bg-zinc-100/70 dark:hover:bg-zinc-800/60'
            "
          >
            {{ tab.label }}
          </button>
        </div>

        <div class="flex-1 min-h-0 p-4 overflow-y-auto">
          <OcrSettingsPreDetectionTab v-if="activeTab === 'preDetection'" :config="config" />
          <OcrSettingsDetectionTab v-else-if="activeTab === 'detection'" :config="config" />
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
            class="text-[11px] text-amber-700 dark:text-amber-400 font-medium truncate"
          >
            {{ strings.modifiedHint }}
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
