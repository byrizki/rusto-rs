<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref } from 'vue';
import type { OcrSettingsTab } from '../../composables/useWebOcrSettings';
import { defaultOcrConfig, type OcrConfig } from '../../types/ocrConfig';
import type { OcrDemoStrings } from '../../types/ocrDemoI18n';
import { webDemoModels, type OcrModelOption } from '../../types/ocrModels';
import { filterSettingsIndex, type OcrSettingSearchEntry } from '../../utils/ocrSettingsIndex';
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
  isCustomConfig: boolean;
  hasRun: boolean;
  isScanning: boolean;
  isInitializing: boolean;
  strings: OcrDemoStrings['settings'];
}>();

const emit = defineEmits<{
  'update:activeTab': [tab: OcrSettingsTab];
  close: [];
  resetDefaults: [];
  selectModel: [modelId: string];
  importConfig: [payload: { modelId?: string; config: Partial<OcrConfig> }];
  layoutModelChanged: [];
  backendChanged: [backend: 'rusto-nn' | 'rusto-rten' | 'rusto-webgpu'];
  applyAndRun: [];
}>();

const tabs = computed<Array<{ id: OcrSettingsTab; label: string }>>(() => [
  { id: 'runtime', label: props.strings.tabRuntime },
  { id: 'detection', label: props.strings.tabDetection },
  { id: 'recognition', label: props.strings.tabRecognition },
]);

const confirmReset = ref(false);
const exportCopied = ref(false);
const showImport = ref(false);
const importText = ref('');
const importError = ref<string | null>(null);
const searchQuery = ref('');
const modalBodyRef = ref<HTMLElement | null>(null);
let activeBeatEl: HTMLElement | null = null;
let activeBeatTimer: ReturnType<typeof setTimeout> | null = null;

const detectionSubTab = ref<'pre' | 'core' | 'post'>('core');
const recognitionSubTab = ref<'pre' | 'core' | 'post'>('core');
const runtimeSubTab = ref<'engines' | 'model' | 'weights'>('engines');

const searchResults = computed<OcrSettingSearchEntry[]>(() =>
  filterSettingsIndex(searchQuery.value)
);

function clearSettingBeat() {
  if (activeBeatTimer !== null) {
    clearTimeout(activeBeatTimer);
    activeBeatTimer = null;
  }
  if (activeBeatEl) {
    activeBeatEl.classList.remove('ocr-setting-beat');
    activeBeatEl = null;
  }
}

function focusAndBeatSetting(settingId: string) {
  const bodyEl = modalBodyRef.value;
  if (!bodyEl) return;

  const fallbackId = settingId === 'lineThresholds' ? 'layoutModel' : null;
  const targetEl =
    bodyEl.querySelector<HTMLElement>(`[data-setting-id="${settingId}"]`) ??
    (fallbackId ? bodyEl.querySelector<HTMLElement>(`[data-setting-id="${fallbackId}"]`) : null);
  if (!targetEl) return;

  clearSettingBeat();
  targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

  const focusable = targetEl.querySelector<HTMLElement>(
    'input:not(:disabled), select:not(:disabled), button:not(:disabled)'
  );
  focusable?.focus({ preventScroll: true });

  void targetEl.offsetWidth;
  targetEl.classList.add('ocr-setting-beat');
  activeBeatEl = targetEl;
  activeBeatTimer = setTimeout(() => {
    clearSettingBeat();
  }, 1500);
}

onBeforeUnmount(() => {
  clearSettingBeat();
});

async function jumpToSearchResult(entry: OcrSettingSearchEntry) {
  if (entry.tab === 'detection') {
    detectionSubTab.value = entry.subTab as 'pre' | 'core' | 'post';
  } else if (entry.tab === 'recognition') {
    recognitionSubTab.value = entry.subTab as 'pre' | 'core' | 'post';
  } else {
    runtimeSubTab.value = entry.subTab as 'engines' | 'model' | 'weights';
  }
  emit('update:activeTab', entry.tab);
  searchQuery.value = '';

  await nextTick();
  await nextTick();
  focusAndBeatSetting(entry.id);
}

function onPrimaryTabKeydown(event: KeyboardEvent, index: number) {
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  event.preventDefault();
  const dir = event.key === 'ArrowRight' ? 1 : -1;
  const nextIndex = (index + dir + tabs.value.length) % tabs.value.length;
  const nextTab = tabs.value[nextIndex];
  if (nextTab) {
    emit('update:activeTab', nextTab.id);
  }
}

function handleResetClick() {
  if (props.isCustomConfig && !confirmReset.value) {
    confirmReset.value = true;
    return;
  }
  confirmReset.value = false;
  emit('resetDefaults');
}

async function handleExportJson() {
  const payload = JSON.stringify(
    {
      modelId: props.selectedModelId,
      config: props.config,
    },
    null,
    2
  );
  try {
    await navigator.clipboard.writeText(payload);
    exportCopied.value = true;
    setTimeout(() => {
      exportCopied.value = false;
    }, 1800);
  } catch {
    showImport.value = true;
    importText.value = payload;
  }
}

function validateConfigFields(
  rawConfig: Record<string, unknown>
): { config: Partial<OcrConfig> } | { error: string } {
  const validated: Partial<OcrConfig> = {};
  const unknownKeys: string[] = [];
  for (const [key, val] of Object.entries(rawConfig)) {
    if (key === 'modelId') continue;
    if (!(key in defaultOcrConfig)) {
      unknownKeys.push(key);
      continue;
    }
    const expectedType = typeof (defaultOcrConfig as unknown as Record<string, unknown>)[key];
    if (typeof val !== expectedType) {
      return { error: `Invalid type for "${key}": expected ${expectedType}, got ${typeof val}.` };
    }
    (validated as Record<string, unknown>)[key] = val;
  }
  if (unknownKeys.length > 0) {
    return { error: `Unsupported keys: ${unknownKeys.join(', ')}` };
  }
  return { config: validated };
}

function handleImportSubmit() {
  importError.value = null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(importText.value);
  } catch (err) {
    importError.value = `Invalid JSON: ${err instanceof Error ? err.message : 'parse error'}`;
    return;
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    importError.value = 'Expected a JSON object with config fields or { modelId, config }.';
    return;
  }

  const record = parsed as Record<string, unknown>;
  const rawConfig =
    record.config && typeof record.config === 'object' && !Array.isArray(record.config)
      ? (record.config as Record<string, unknown>)
      : record;

  let modelId: string | undefined;
  if (typeof record.modelId === 'string') {
    if (!webDemoModels.some((m) => m.id === record.modelId)) {
      importError.value = `Unsupported modelId: "${record.modelId}".`;
      return;
    }
    modelId = record.modelId;
  }

  const result = validateConfigFields(rawConfig);
  if ('error' in result) {
    importError.value = result.error;
    return;
  }

  emit('importConfig', { modelId, config: result.config });
  showImport.value = false;
  importText.value = '';
}
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
        class="w-full max-w-[600px] max-h-[88vh] flex flex-col rounded-xl shadow-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs overflow-hidden"
      >
        <!-- Sticky Header -->
        <div
          class="flex items-center justify-between gap-2 px-4 py-2.5 border-b border-zinc-200/80 dark:border-zinc-800 shrink-0"
        >
          <div class="flex items-center gap-2 min-w-0">
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
            <h3 class="text-sm font-semibold text-zinc-900 dark:text-zinc-100 truncate">
              {{ strings.title }}
            </h3>
          </div>

          <div class="flex items-center gap-1 shrink-0">
            <button
              type="button"
              @click="handleExportJson"
              :title="exportCopied ? 'Copied config JSON' : 'Export config JSON to clipboard'"
              aria-label="Export config JSON"
              class="px-2 py-1 rounded-md text-[11px] font-mono text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors cursor-pointer"
            >
              {{ exportCopied ? 'Copied!' : 'Export' }}
            </button>
            <button
              type="button"
              @click="showImport = !showImport"
              title="Import config JSON"
              aria-label="Import config JSON"
              class="px-2 py-1 rounded-md text-[11px] font-mono transition-colors cursor-pointer"
              :class="
                showImport
                  ? 'bg-primary/10 text-primary font-semibold'
                  : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-800'
              "
            >
              Import
            </button>

            <div class="w-px h-4 bg-zinc-200 dark:bg-zinc-800 mx-0.5" />

            <template v-if="confirmReset">
              <button
                type="button"
                @click="handleResetClick"
                class="px-2 py-1 rounded-md text-[11px] font-semibold bg-red-500/10 text-red-600 dark:text-red-400 hover:bg-red-500/20 transition-colors cursor-pointer"
              >
                Confirm Reset
              </button>
              <button
                type="button"
                @click="confirmReset = false"
                class="px-1.5 py-1 rounded-md text-[11px] text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                Cancel
              </button>
            </template>
            <button
              v-else
              type="button"
              @click="handleResetClick"
              :title="strings.resetDefaults"
              :aria-label="strings.resetDefaults"
              class="p-1.5 rounded-md text-zinc-400 hover:text-primary hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors cursor-pointer"
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
              title="Close (Esc)"
              aria-label="Close settings"
              class="p-1.5 rounded-md text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors cursor-pointer"
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

        <!-- Inline JSON Import Drawer -->
        <div
          v-if="showImport"
          class="px-4 py-2.5 border-b border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 space-y-2 shrink-0"
        >
          <div class="flex items-center justify-between">
            <span class="text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">
              Import Configuration JSON
            </span>
            <span class="text-[10px] font-mono text-zinc-400">
              { "modelId"?: string, "config": { ... } }
            </span>
          </div>
          <textarea
            v-model="importText"
            rows="3"
            placeholder="Paste exported JSON config here..."
            class="w-full p-2 text-[11px] font-mono rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-primary"
          />
          <p v-if="importError" class="text-[11px] text-red-600 dark:text-red-400 font-mono">
            {{ importError }}
          </p>
          <div class="flex justify-end gap-2">
            <button
              type="button"
              @click="showImport = false"
              class="px-2.5 py-1 rounded-md text-[11px] border border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              @click="handleImportSubmit"
              class="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-primary text-white cursor-pointer"
            >
              Apply JSON
            </button>
          </div>
        </div>

        <!-- Primary Navigation & Search -->
        <div
          class="px-4 py-2 border-b border-zinc-200 dark:border-zinc-800 bg-zinc-100/60 dark:bg-zinc-950/80 space-y-2 shrink-0"
        >
          <div class="relative">
            <svg
              class="w-3.5 h-3.5 text-zinc-400 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </svg>
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search options (e.g. unclip, descreen, batch, WebGPU, limitSideLen)..."
              aria-label="Search OCR settings"
              class="w-full pl-8 pr-7 py-1 text-[11px] rounded-md border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 focus:outline-none focus:border-primary"
            />
            <button
              v-if="searchQuery"
              type="button"
              @click="searchQuery = ''"
              aria-label="Clear search"
              class="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 cursor-pointer"
            >
              ×
            </button>
          </div>

          <div
            v-if="!searchQuery.trim()"
            class="grid grid-cols-3 gap-1 p-1 rounded-lg bg-zinc-200/60 dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800"
            role="tablist"
            aria-label="Pipeline Settings Sections"
          >
            <button
              v-for="(tab, idx) in tabs"
              :key="tab.id"
              type="button"
              role="tab"
              :aria-selected="activeTab === tab.id"
              :tabindex="activeTab === tab.id ? 0 : -1"
              @click="emit('update:activeTab', tab.id)"
              @keydown="onPrimaryTabKeydown($event, idx)"
              class="inline-flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-md text-xs font-medium transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
              :class="
                activeTab === tab.id
                  ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 font-semibold shadow-2xs ring-1 ring-zinc-200/80 dark:ring-zinc-700'
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

        <!-- Scrollable Body -->
        <div ref="modalBodyRef" class="flex-1 min-h-0 px-4 py-3 overflow-y-auto">
          <div v-if="searchQuery.trim()" class="space-y-1.5">
            <div class="flex items-center justify-between text-[10px] font-mono text-zinc-400 pb-1">
              <span>{{ searchResults.length }} matching option(s)</span>
              <span>Click an option to open its stage</span>
            </div>

            <div v-if="searchResults.length === 0" class="py-8 text-center text-zinc-400 text-xs">
              No settings match "{{ searchQuery.trim() }}".
            </div>

            <button
              v-for="entry in searchResults"
              :key="entry.id"
              type="button"
              @click="jumpToSearchResult(entry)"
              class="w-full text-left p-2.5 rounded-lg border border-zinc-200/80 dark:border-zinc-800 bg-white dark:bg-zinc-900 hover:border-primary/60 hover:bg-primary/5 transition-colors cursor-pointer space-y-1"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="flex items-center gap-1.5 min-w-0 flex-wrap">
                  <span class="font-medium text-zinc-900 dark:text-zinc-100 text-xs">
                    {{ entry.label }}
                  </span>
                  <code
                    class="text-[10px] font-mono text-zinc-500 dark:text-zinc-400 bg-zinc-100 dark:bg-zinc-800 px-1 py-0.5 rounded"
                  >
                    {{ entry.identifier }}
                  </code>
                </div>
                <span
                  class="shrink-0 font-mono text-[10px] px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300"
                >
                  {{ entry.getValue(config) }}
                </span>
              </div>
              <div class="flex items-center justify-between gap-2">
                <p class="text-[11px] text-zinc-500 dark:text-zinc-400 truncate">
                  {{ entry.description }}
                </p>
                <span class="shrink-0 text-[10px] font-mono text-primary">
                  {{ entry.breadcrumb }} →
                </span>
              </div>
            </button>
          </div>

          <KeepAlive v-else>
            <OcrSettingsRuntimeTab
              v-if="activeTab === 'runtime'"
              v-model:sub-tab="runtimeSubTab"
              :config="config"
              :current-model="currentModel"
              :selected-model-id="selectedModelId"
              :base-u-r-l="baseURL"
              :is-web-gpu-supported="isWebGpuSupported"
              @select-model="(modelId) => emit('selectModel', modelId)"
              @backend-changed="(b) => emit('backendChanged', b)"
            />
            <OcrSettingsDetectionTab
              v-else-if="activeTab === 'detection'"
              v-model:sub-tab="detectionSubTab"
              :config="config"
            />
            <OcrSettingsRecognitionTab
              v-else
              v-model:sub-tab="recognitionSubTab"
              :config="config"
              :current-model="currentModel"
              @layout-model-changed="emit('layoutModelChanged')"
            />
          </KeepAlive>
        </div>

        <!-- Sticky Footer -->
        <div
          class="px-4 py-2.5 border-t border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/80 dark:bg-zinc-950/60 flex items-center justify-between gap-2 shrink-0"
        >
          <div class="flex items-center gap-1.5 text-[11px] min-w-0">
            <span
              v-if="isScanning || isInitializing"
              class="inline-flex items-center gap-1.5 text-primary font-medium"
            >
              <span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              {{ isInitializing ? 'Loading model...' : 'Running OCR...' }}
            </span>
            <span
              v-else-if="isConfigModified"
              class="inline-flex items-center gap-1.5 text-amber-600 dark:text-amber-400 truncate"
              :title="strings.modifiedHint"
            >
              <span class="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0" />
              <span class="truncate">{{ strings.modifiedHint }}</span>
            </span>
            <span v-else class="text-zinc-400 font-mono text-[10px]">
              {{ selectedModelId }} · {{ config.inferenceBackend }}
            </span>
          </div>

          <div class="flex items-center gap-2 ml-auto shrink-0">
            <button
              type="button"
              @click="emit('close')"
              class="px-3 py-1.5 rounded-lg text-xs font-medium border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors cursor-pointer"
            >
              {{ strings.close }}
            </button>
            <button
              type="button"
              :disabled="isScanning || isInitializing"
              @click="emit('applyAndRun')"
              title="Apply settings and run OCR (Ctrl+Enter)"
              class="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-primary text-white hover:opacity-95 active:scale-[0.98] transition-all cursor-pointer shadow-xs disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {{ isScanning ? 'Scanning...' : strings.applyAndRun }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style>
@keyframes ocr-setting-border-beat {
  0%,
  100% {
    outline: 2px solid rgb(249 115 22 / 0);
    outline-offset: 4px;
    background-color: transparent;
  }
  16%,
  52% {
    outline: 2px solid rgb(249 115 22 / 0.9);
    outline-offset: 4px;
    background-color: rgb(249 115 22 / 0.06);
  }
  34% {
    outline: 2px solid rgb(249 115 22 / 0.25);
    outline-offset: 2px;
    background-color: transparent;
  }
  78% {
    outline: 2px solid rgb(249 115 22 / 0.45);
    outline-offset: 4px;
    background-color: rgb(249 115 22 / 0.02);
  }
}

.ocr-setting-beat {
  border-radius: 0.375rem;
  animation: ocr-setting-border-beat 1.45s ease-out 1;
}
</style>
