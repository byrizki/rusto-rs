<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import type { TextResult } from '@rustors/web';
import type { ActiveRunConfig, OcrConfig } from '../../types/ocrConfig';
import type { OcrDemoStrings } from '../../types/ocrDemoI18n';
import type { OcrModelOption } from '../../types/ocrModels';
import {
  downloadTextFile,
  formatFluentBuilderSnippet,
  formatResultsTsv,
} from '../../utils/ocrResultFormatter';
import { buildDetectTextOptions } from '../../types/ocrConfig';
import OcrActiveConfigView from '../OcrActiveConfigView.vue';

export type OcrOutputTab = 'structured' | 'spatial' | 'tsv' | 'json' | 'config';

const props = defineProps<{
  results: TextResult[];
  spatialText: string;
  outputMode: 'lines' | 'words';
  isScanning: boolean;
  ocrError: string | null;
  hoveredIndex: number | null;
  activeRunConfig: ActiveRunConfig | null;
  ocrConfig: OcrConfig;
  currentModel: OcrModelOption;
  isConfigModified: boolean;
  strings: OcrDemoStrings['results'];
}>();

const emit = defineEmits<{
  'update:hoveredIndex': [index: number | null];
  changeGranularity: [mode: 'lines' | 'words'];
  dismissError: [];
  openSettings: [];
  runOcr: [];
}>();

const activeTab = ref<OcrOutputTab>('structured');
const tsvViewMode = ref<'table' | 'raw'>('table');
const searchQuery = ref<string>('');
const copiedAll = ref<boolean>(false);
const copiedRowIdx = ref<number | null>(null);
const isPointerInsideList = ref<boolean>(false);
const listContainerRef = ref<HTMLElement | null>(null);

const hasRun = computed<boolean>(() => props.activeRunConfig !== null);

const filteredEntries = computed(() => {
  const q = searchQuery.value.trim().toLowerCase();
  const indexed = props.results.map((item, idx) => ({ item, idx }));
  if (!q) return indexed;
  return indexed.filter(({ item }) => item.text.toLowerCase().includes(q));
});

const tsvExport = computed(() => formatResultsTsv(props.results));
const jsonExport = computed(() => JSON.stringify(props.results, null, 2));

const activeScorePercent = computed(() => {
  const score =
    props.activeRunConfig?.options.postRecognition?.scoreThreshold ??
    props.activeRunConfig?.rawConfig.textScore ??
    props.ocrConfig.textScore;
  return Math.round(score * 100);
});

function getActiveExportPayload(): { content: string; filename: string; mime: string } {
  if (activeTab.value === 'spatial') {
    return { content: props.spatialText, filename: 'rusto-spatial.txt', mime: 'text/plain' };
  }
  if (activeTab.value === 'tsv') {
    return {
      content: tsvExport.value,
      filename: 'rusto-results.tsv',
      mime: 'text/tab-separated-values',
    };
  }
  if (activeTab.value === 'json') {
    return { content: jsonExport.value, filename: 'rusto-results.json', mime: 'application/json' };
  }
  if (activeTab.value === 'config') {
    const opts =
      props.activeRunConfig?.options ?? buildDetectTextOptions(props.ocrConfig, props.outputMode);
    return {
      content: formatFluentBuilderSnippet(props.outputMode, opts),
      filename: 'rusto-snippet.ts',
      mime: 'text/typescript',
    };
  }
  return {
    content: props.results.map((r) => r.text).join('\n'),
    filename: 'rusto-text.txt',
    mime: 'text/plain',
  };
}

async function copyAllActive() {
  const { content } = getActiveExportPayload();
  if (!content) return;
  try {
    await navigator.clipboard.writeText(content);
    copiedAll.value = true;
    setTimeout(() => {
      copiedAll.value = false;
    }, 1800);
  } catch (err) {
    console.error('Copy failed:', err);
  }
}

async function copyRowText(text: string, idx: number) {
  try {
    await navigator.clipboard.writeText(text);
    copiedRowIdx.value = idx;
    setTimeout(() => {
      if (copiedRowIdx.value === idx) copiedRowIdx.value = null;
    }, 1500);
  } catch (err) {
    console.error('Row copy failed:', err);
  }
}

function handleDownload() {
  const { content, filename, mime } = getActiveExportPayload();
  if (!content) return;
  downloadTextFile(content, filename, mime);
}

watch(
  () => props.hoveredIndex,
  (idx) => {
    if (idx === null || isPointerInsideList.value || !listContainerRef.value) return;
    nextTick(() => {
      const el = listContainerRef.value?.querySelector<HTMLElement>(`[data-ocr-idx="${idx}"]`);
      el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
  }
);

watch(
  () => props.results,
  () => {
    searchQuery.value = '';
  }
);
</script>

<template>
  <aside
    class="lg:col-span-5 flex flex-col bg-white dark:bg-zinc-900 h-full min-h-[380px] lg:min-h-0 overflow-hidden"
  >
    <div
      class="flex items-center border-b border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/60 px-2 text-xs overflow-x-auto shrink-0"
      role="tablist"
    >
      <button
        type="button"
        role="tab"
        :aria-selected="activeTab === 'structured'"
        @click="activeTab = 'structured'"
        class="inline-flex items-center gap-1.5 px-2.5 py-2.5 border-b-2 font-medium transition-colors cursor-pointer whitespace-nowrap"
        :class="
          activeTab === 'structured'
            ? 'border-primary text-primary'
            : 'border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
        "
      >
        <span>{{ strings.tabResults }}</span>
        <span
          v-if="hasRun || isScanning"
          class="px-1.5 py-0.2 rounded-full text-[10px] font-mono tabular-nums"
          :class="
            activeTab === 'structured'
              ? 'bg-primary/15 text-primary'
              : 'bg-zinc-200/80 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400'
          "
        >
          {{ isScanning ? '…' : results.length }}
        </span>
      </button>

      <button
        type="button"
        role="tab"
        :aria-selected="activeTab === 'spatial'"
        @click="activeTab = 'spatial'"
        class="px-2.5 py-2.5 border-b-2 font-medium transition-colors cursor-pointer whitespace-nowrap"
        :class="
          activeTab === 'spatial'
            ? 'border-primary text-primary'
            : 'border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
        "
      >
        {{ strings.tabSpatial }}
      </button>

      <button
        type="button"
        role="tab"
        :aria-selected="activeTab === 'tsv'"
        @click="activeTab = 'tsv'"
        class="px-2.5 py-2.5 border-b-2 font-medium transition-colors cursor-pointer whitespace-nowrap"
        :class="
          activeTab === 'tsv'
            ? 'border-primary text-primary'
            : 'border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
        "
      >
        {{ strings.tabTable }}
      </button>

      <button
        type="button"
        role="tab"
        :aria-selected="activeTab === 'json'"
        @click="activeTab = 'json'"
        class="px-2.5 py-2.5 border-b-2 font-medium transition-colors cursor-pointer whitespace-nowrap"
        :class="
          activeTab === 'json'
            ? 'border-primary text-primary'
            : 'border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
        "
      >
        {{ strings.tabJson }}
      </button>

      <button
        type="button"
        role="tab"
        :aria-selected="activeTab === 'config'"
        @click="activeTab = 'config'"
        class="inline-flex items-center gap-1.5 px-2.5 py-2.5 border-b-2 font-medium transition-colors cursor-pointer whitespace-nowrap"
        :class="
          activeTab === 'config'
            ? 'border-primary text-primary'
            : 'border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
        "
      >
        <span>{{ strings.tabCode }}</span>
        <span
          v-if="isConfigModified"
          class="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0"
          :title="strings.footerModified"
        />
      </button>
    </div>

    <div
      v-if="hasRun || activeTab === 'config'"
      class="px-3 py-2 border-b border-zinc-200/70 dark:border-zinc-800/80 flex flex-wrap items-center justify-between gap-2 shrink-0 bg-white dark:bg-zinc-900"
    >
      <div
        class="inline-flex items-center p-0.5 rounded-lg bg-zinc-100 dark:bg-zinc-950 border border-zinc-200/80 dark:border-zinc-800"
      >
        <button
          type="button"
          @click="emit('changeGranularity', 'lines')"
          class="px-2.5 py-0.5 text-[11px] rounded-md font-medium transition-all cursor-pointer"
          :class="
            outputMode === 'lines'
              ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-2xs'
              : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
          "
        >
          {{ strings.modeLines }}
        </button>
        <button
          type="button"
          @click="emit('changeGranularity', 'words')"
          class="px-2.5 py-0.5 text-[11px] rounded-md font-medium transition-all cursor-pointer"
          :class="
            outputMode === 'words'
              ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-2xs'
              : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
          "
        >
          {{ strings.modeWords }}
        </button>
      </div>

      <div
        v-if="hasRun && results.length > 4 && (activeTab === 'structured' || activeTab === 'tsv')"
        class="relative flex-1 min-w-[120px] max-w-[180px]"
      >
        <input
          v-model="searchQuery"
          type="search"
          :placeholder="strings.searchPlaceholder"
          class="w-full pl-2.5 pr-2 py-1 text-[11px] rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 focus:outline-none focus:border-primary"
        />
      </div>

      <div class="inline-flex items-center gap-1.5 ml-auto">
        <button
          type="button"
          :disabled="!hasRun && activeTab !== 'config'"
          @click="copyAllActive"
          class="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 disabled:opacity-40 transition-colors cursor-pointer"
        >
          <span>{{ copiedAll ? `✓ ${strings.copied}` : strings.copy }}</span>
        </button>
        <button
          type="button"
          :disabled="!hasRun && activeTab !== 'config'"
          @click="handleDownload"
          class="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 disabled:opacity-40 transition-colors cursor-pointer"
          :title="strings.download"
        >
          <svg
            class="w-3.5 h-3.5 shrink-0"
            viewBox="0 0 20 20"
            fill="currentColor"
            aria-hidden="true"
          >
            <path
              d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z"
            />
            <path
              d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z"
            />
          </svg>
          <span class="hidden sm:inline">{{ strings.download }}</span>
        </button>
      </div>
    </div>

    <div
      ref="listContainerRef"
      class="flex-1 min-h-0 p-3 overflow-y-auto max-h-[460px] lg:max-h-full text-xs"
      @mouseenter="isPointerInsideList = true"
      @mouseleave="isPointerInsideList = false"
    >
      <div
        v-if="ocrError"
        class="mb-3 p-3 rounded-xl border border-red-200 dark:border-red-900/80 bg-red-50 dark:bg-red-950/50 text-red-700 dark:text-red-300 flex items-start gap-2.5"
        role="alert"
      >
        <svg
          class="w-4 h-4 text-red-500 shrink-0 mt-0.5"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fill-rule="evenodd"
            d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-8-5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 10a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
            clip-rule="evenodd"
          />
        </svg>
        <pre
          class="flex-1 min-w-0 text-[11px] font-mono whitespace-pre-wrap break-words select-text"
          >{{ ocrError }}</pre>
        <button
          type="button"
          @click="emit('dismissError')"
          class="text-red-400 hover:text-red-600 dark:hover:text-red-200 p-0.5 cursor-pointer"
          aria-label="Dismiss error"
        >
          ✕
        </button>
      </div>

      <template v-if="isScanning">
        <div class="space-y-2">
          <div
            v-for="i in 6"
            :key="i"
            class="p-2.5 rounded-lg border border-zinc-200/70 dark:border-zinc-800/70 bg-zinc-50 dark:bg-zinc-900/60 flex items-center justify-between gap-3 animate-pulse"
          >
            <div
              class="h-3.5 rounded bg-zinc-200 dark:bg-zinc-800"
              :style="{ width: `${38 + ((i * 17) % 48)}%` }"
            />
            <div class="h-4 w-10 rounded bg-zinc-200 dark:bg-zinc-800 shrink-0" />
          </div>
        </div>
      </template>

      <template v-else-if="!hasRun && activeTab !== 'config'">
        <div
          class="h-full min-h-[260px] flex flex-col items-center justify-center text-center px-5 py-8"
        >
          <div
            class="w-11 h-11 rounded-2xl bg-primary/10 text-primary border border-primary/20 flex items-center justify-center mb-3.5"
          >
            <svg
              class="w-5 h-5"
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
              <path d="M7 9h10" />
              <path d="M7 13h7" />
            </svg>
          </div>
          <h4 class="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            {{ strings.idleTitle }}
          </h4>
          <p class="text-xs text-zinc-500 dark:text-zinc-400 mt-1 max-w-[270px] leading-relaxed">
            {{ strings.idleDescription }}
          </p>
          <button
            type="button"
            @click="emit('runOcr')"
            class="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-primary text-white hover:opacity-95 active:scale-[0.98] transition-all cursor-pointer shadow-sm"
          >
            <svg class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path
                d="M6.3 2.84A1.5 1.5 0 0 0 4 4.11v11.78a1.5 1.5 0 0 0 2.3 1.27l9.344-5.891a1.5 1.5 0 0 0 0-2.538L6.3 2.841Z"
              />
            </svg>
            <span>{{ strings.idleAction }}</span>
          </button>
          <p class="mt-3 text-[11px] text-zinc-400 dark:text-zinc-500">
            {{ strings.idlePrivacyNote }}
          </p>
        </div>
      </template>

      <template v-else>
        <div v-if="activeTab === 'structured'" class="space-y-1.5">
          <div
            v-for="{ item, idx } in filteredEntries"
            :key="idx"
            :data-ocr-idx="idx"
            @mouseenter="emit('update:hoveredIndex', idx)"
            @mouseleave="emit('update:hoveredIndex', null)"
            @click="copyRowText(item.text, idx)"
            class="group p-2 rounded-lg border transition-colors flex items-start justify-between gap-2.5 cursor-pointer"
            :class="
              hoveredIndex === idx
                ? 'border-primary bg-primary/8 dark:bg-primary/15'
                : 'border-zinc-200/80 dark:border-zinc-800/90 bg-zinc-50/40 dark:bg-zinc-950/30 hover:border-zinc-300 dark:hover:border-zinc-700'
            "
          >
            <span
              class="text-[10px] font-mono text-zinc-400 dark:text-zinc-500 pt-0.5 w-5 shrink-0 tabular-nums select-none"
            >
              {{ idx + 1 }}
            </span>
            <div
              class="flex-1 min-w-0 text-zinc-800 dark:text-zinc-200 leading-snug break-words select-text"
            >
              {{ item.text }}
            </div>
            <span
              v-if="copiedRowIdx === idx"
              class="inline-flex items-center justify-center w-6 h-5 rounded bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 shrink-0"
              :title="strings.copied"
              :aria-label="strings.copied"
            >
              <svg
                class="w-3.5 h-3.5"
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
            </span>
            <span
              v-else
              class="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold shrink-0 tabular-nums"
              :class="
                item.score >= 0.9
                  ? 'bg-emerald-100/80 text-emerald-800 dark:bg-emerald-950/70 dark:text-emerald-400'
                  : item.score >= 0.75
                    ? 'bg-sky-100/80 text-sky-800 dark:bg-sky-950/70 dark:text-sky-400'
                    : 'bg-amber-100/80 text-amber-800 dark:bg-amber-950/70 dark:text-amber-400'
              "
            >
              {{ Math.round(item.score * 100) }}%
            </span>
          </div>

          <div v-if="results.length === 0" class="py-10 px-4 text-center space-y-2">
            <p class="text-zinc-500 dark:text-zinc-400">{{ strings.emptyNoMatchTitle }}</p>
            <button
              type="button"
              @click="emit('openSettings')"
              class="text-primary hover:underline font-medium cursor-pointer"
            >
              {{ strings.emptyNoMatchAction }}
            </button>
          </div>
          <p v-else-if="filteredEntries.length === 0" class="py-8 px-4 text-center text-zinc-400">
            {{ strings.emptyFilteredTitle }}
          </p>
        </div>

        <pre
          v-else-if="activeTab === 'spatial'"
          class="whitespace-pre overflow-x-auto text-zinc-800 dark:text-zinc-200 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 font-mono text-[11px] leading-relaxed select-text"
          >{{ spatialText }}</pre>

        <div v-else-if="activeTab === 'tsv'" class="space-y-2">
          <div class="flex items-center justify-between text-[11px]">
            <div
              class="inline-flex items-center p-0.5 rounded-md bg-zinc-100 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800"
            >
              <button
                type="button"
                @click="tsvViewMode = 'table'"
                class="px-2 py-0.5 rounded transition-colors cursor-pointer"
                :class="
                  tsvViewMode === 'table'
                    ? 'bg-white dark:bg-zinc-800 font-medium text-zinc-900 dark:text-zinc-100 shadow-2xs'
                    : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                "
              >
                {{ strings.tableViewGrid }}
              </button>
              <button
                type="button"
                @click="tsvViewMode = 'raw'"
                class="px-2 py-0.5 rounded transition-colors cursor-pointer"
                :class="
                  tsvViewMode === 'raw'
                    ? 'bg-white dark:bg-zinc-800 font-medium text-zinc-900 dark:text-zinc-100 shadow-2xs'
                    : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                "
              >
                {{ strings.tableViewRaw }}
              </button>
            </div>
            <span class="text-[10px] text-zinc-400 hidden sm:inline">
              {{ strings.tableHint }}
            </span>
          </div>

          <div
            v-if="tsvViewMode === 'table'"
            class="overflow-x-auto border border-zinc-200 dark:border-zinc-800 rounded-lg"
          >
            <table class="w-full text-left text-[11px] border-collapse">
              <thead>
                <tr
                  class="bg-zinc-50 dark:bg-zinc-950 border-b border-zinc-200 dark:border-zinc-800 text-zinc-500 dark:text-zinc-400 font-semibold text-[10px]"
                >
                  <th class="py-1.5 px-2 w-8">#</th>
                  <th class="py-1.5 px-2">{{ strings.colText }}</th>
                  <th class="py-1.5 px-2 w-16">{{ strings.colScore }}</th>
                  <th class="py-1.5 px-2 w-32">{{ strings.colBox }}</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-zinc-100 dark:divide-zinc-800/70">
                <tr
                  v-for="{ item: row, idx } in filteredEntries"
                  :key="idx"
                  :data-ocr-idx="idx"
                  @mouseenter="emit('update:hoveredIndex', idx)"
                  @mouseleave="emit('update:hoveredIndex', null)"
                  @click="copyRowText(row.text, idx)"
                  class="transition-colors cursor-pointer hover:bg-primary/5 dark:hover:bg-primary/10"
                  :class="hoveredIndex === idx ? 'bg-primary/10 dark:bg-primary/20' : ''"
                >
                  <td class="py-1.5 px-2 text-zinc-400 font-mono text-[10px] tabular-nums">
                    {{ idx + 1 }}
                  </td>
                  <td class="py-1.5 px-2 font-medium text-zinc-800 dark:text-zinc-200 select-text">
                    {{ row.text }}
                  </td>
                  <td class="py-1.5 px-2 font-mono text-[10px] tabular-nums">
                    {{ (row.score * 100).toFixed(1) }}%
                  </td>
                  <td
                    class="py-1.5 px-2 text-zinc-500 dark:text-zinc-400 font-mono text-[10px] tabular-nums"
                  >
                    {{ Math.round(row.frame.left) }}, {{ Math.round(row.frame.top) }},
                    {{ Math.round(row.frame.width) }}, {{ Math.round(row.frame.height) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <pre
            v-else
            class="whitespace-pre overflow-x-auto text-zinc-700 dark:text-zinc-300 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 font-mono text-[11px] leading-relaxed select-text"
            >{{ tsvExport }}</pre>
        </div>

        <pre
          v-else-if="activeTab === 'json'"
          class="whitespace-pre overflow-x-auto text-zinc-700 dark:text-zinc-300 p-3 rounded-lg bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 font-mono text-[11px] leading-relaxed select-text"
          >{{ jsonExport }}</pre>

        <OcrActiveConfigView
          v-else-if="activeTab === 'config'"
          :active-run="activeRunConfig"
          :current-ocr-config="ocrConfig"
          :current-model-name="currentModel.name"
          :current-output-mode="outputMode"
          :is-modified-since-run="isConfigModified"
        />
      </template>
    </div>

    <footer
      class="mt-auto px-3 py-1.5 border-t border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/70 flex items-center justify-between gap-2 text-[11px] shrink-0"
    >
      <div class="flex items-center gap-1.5 min-w-0">
        <span
          v-if="activeRunConfig"
          class="truncate font-mono text-[10px] text-zinc-600 dark:text-zinc-300 tabular-nums"
        >
          {{ activeRunConfig.outputMode === 'words' ? 'asWords()' : 'asLines()' }} • ≥{{
            activeScorePercent
          }}%
        </span>
        <span v-else class="text-zinc-400 dark:text-zinc-500 text-[11px]">
          {{ strings.footerNoRun }}
        </span>

        <span
          v-if="isConfigModified"
          class="inline-flex items-center justify-center w-4 h-4 rounded-full bg-amber-500/15 text-amber-600 dark:text-amber-400 shrink-0"
          :title="strings.footerModified"
          :aria-label="strings.footerModified"
        >
          <span class="w-1.5 h-1.5 rounded-full bg-amber-500" />
        </span>
      </div>

      <button
        v-if="activeTab !== 'config'"
        type="button"
        @click="activeTab = 'config'"
        class="text-primary hover:underline font-medium shrink-0 cursor-pointer text-[11px]"
      >
        {{ strings.footerViewCode }}
      </button>
    </footer>
  </aside>
</template>
