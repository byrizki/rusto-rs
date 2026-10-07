<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { useRuntimeConfig } from '#app';
import {
  initialize,
  detectText,
  isInitialized,
  formatSpatialText,
  groupCandidatesIntoLines,
  type TextResult,
  type OutputGranularity,
} from 'rusto-web';

const runtimeConfig = useRuntimeConfig();
const rawBaseURL = runtimeConfig.app.baseURL || '/';
const baseURL = rawBaseURL.endsWith('/') ? rawBaseURL : `${rawBaseURL}/`;

// Sample images configuration
const samples = [
  {
    id: 'invoice',
    name: 'Invoice / Receipt',
    description: 'Structured layout with numbers & tables',
    url: `${baseURL}samples/invoice1.jpg`,
  },
  {
    id: 'document',
    name: 'General Document',
    description: 'Multi-line text document scan',
    url: `${baseURL}samples/example1.png`,
  },
];

// Reactive state
const selectedSample = ref<string>('invoice');
const imageSrc = ref<string>(samples[0].url);
const status = ref<'idle' | 'initializing' | 'ready' | 'error'>('idle');
const statusMessage = ref<string>('Loading models...');
const isScanning = ref<boolean>(false);
const durationMs = ref<number | null>(null);

// Configuration state
const outputMode = ref<OutputGranularity>('lines');
const textScoreThreshold = ref<number>(0.5);
const activeTab = ref<'structured' | 'spatial' | 'csv' | 'json'>('structured');
const copied = ref<boolean>(false);

// Results state
const results = ref<TextResult[]>([]);
const spatialText = ref<string>('');
const hoveredIndex = ref<number | null>(null);

// Image scaling state
const imgRef = ref<HTMLImageElement | null>(null);
const naturalWidth = ref<number>(1);
const naturalHeight = ref<number>(1);
const renderWidth = ref<number>(1);
const renderHeight = ref<number>(1);

const scaleX = computed(() => renderWidth.value / naturalWidth.value || 1);
const scaleY = computed(() => renderHeight.value / naturalHeight.value || 1);

// Initialize engine
async function initEngine() {
  if (isInitialized()) {
    status.value = 'ready';
    return;
  }
  try {
    status.value = 'initializing';
    statusMessage.value = 'Loading PP-OCRv6 Tiny models...';

    await initialize({
      preset: 'ppv6',
      wasmUrl: `${baseURL}wasm/rusto_rten_wasm_bg.wasm`,
      models: {
        detection: `${baseURL}models/ppocrv6-tiny/det.rten`,
        recognition: `${baseURL}models/ppocrv6-tiny/rec.rten`,
        dictionary: `${baseURL}models/ppocrv6-tiny/dict.txt`,
      },
    });

    status.value = 'ready';
    statusMessage.value = 'Engine Ready';
  } catch (err: unknown) {
    status.value = 'error';
    statusMessage.value = err instanceof Error ? err.message : String(err);
    console.error('OCR Init Error:', err);
  }
}

// Switch granularity and align active inspector tab
function setGranularity(mode: OutputGranularity) {
  outputMode.value = mode;
  if (mode === 'spatial') {
    activeTab.value = 'spatial';
  } else {
    activeTab.value = 'structured';
  }
  runOcr();
}

// Clear results helper
function resetResults() {
  results.value = [];
  spatialText.value = '';
  durationMs.value = null;
  hoveredIndex.value = null;
}

// Run OCR detection
async function runOcr() {
  if (status.value !== 'ready') return;
  resetResults();
  isScanning.value = true;
  // Yield to browser event loop so scanning state and animation paint immediately
  await new Promise<void>((resolve) => {
    setTimeout(resolve, 20);
  });
  const startTime = performance.now();

  try {
    const isSpatial = outputMode.value === 'spatial';
    const opts = {
      output: (isSpatial ? 'lines' : outputMode.value) as OutputGranularity,
      textScore: textScoreThreshold.value,
      calibration: { enabled: true, descreenBackground: true },
      optimization: { enabled: true },
    };

    const output = await detectText(imageSrc.value, opts);
    durationMs.value = Math.round(performance.now() - startTime);

    if (Array.isArray(output)) {
      results.value = output;
      spatialText.value = formatSpatialText(groupCandidatesIntoLines(output));
    } else {
      spatialText.value = output;
      results.value = [];
    }

    if (isSpatial) {
      activeTab.value = 'spatial';
    }
  } catch (err) {
    console.error('OCR Detection error:', err);
  } finally {
    isScanning.value = false;
  }
}

// Handle sample pick
function pickSample(sample: (typeof samples)[0]) {
  resetResults();
  selectedSample.value = sample.id;
  imageSrc.value = sample.url;
}

// Handle custom file upload
function handleFileUpload(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  resetResults();
  selectedSample.value = 'custom';
  imageSrc.value = `${URL.createObjectURL(file)}#${encodeURIComponent(file.name)}`;
}

// Handle image load event to record natural & displayed dimensions
function onImageLoad(e: Event) {
  const img = e.target as HTMLImageElement;
  naturalWidth.value = img.naturalWidth || 1;
  naturalHeight.value = img.naturalHeight || 1;
  renderWidth.value = img.clientWidth || img.naturalWidth || 1;
  renderHeight.value = img.clientHeight || img.naturalHeight || 1;

  // Auto-scan once loaded and ready
  if (status.value === 'ready') {
    runOcr();
  }
}

// Copy results to clipboard
async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    copied.value = true;
    setTimeout(() => {
      copied.value = false;
    }, 2000);
  } catch (err) {
    console.error('Copy failed:', err);
  }
}

// Computed CSV
const csvExport = computed(() => {
  const header = 'text,score,left,top,width,height';
  const rows = results.value.map(
    (r) =>
      `"${r.text.replaceAll('"', '""')}",${r.score.toFixed(4)},${r.frame.left},${r.frame.top},${r.frame.width},${r.frame.height}`
  );
  return [header, ...rows].join('\n');
});

// Computed JSON
const jsonExport = computed(() => JSON.stringify(results.value, null, 2));

onMounted(async () => {
  await initEngine();
});

watch(status, (newStatus) => {
  if (newStatus === 'ready') {
    runOcr();
  }
});
</script>

<template>
  <div
    class="rusto-demo-container not-prose my-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 shadow-xl overflow-hidden"
  >
    <!-- Top Action & Status Bar -->
    <div
      class="demo-header flex flex-wrap items-center justify-between gap-4 p-4 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950"
    >
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2">
          <span
            class="inline-block w-2.5 h-2.5 rounded-full"
            :class="{
              'bg-emerald-500 animate-pulse': status === 'ready',
              'bg-amber-500 animate-pulse': status === 'initializing',
              'bg-red-500': status === 'error',
              'bg-neutral-400': status === 'idle',
            }"
          />
          <span
            class="text-xs font-semibold tracking-wider uppercase text-neutral-600 dark:text-neutral-300"
          >
            {{ statusMessage }}
          </span>
        </div>
        <span
          class="text-xs px-2 py-0.5 rounded-full bg-neutral-200 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 font-mono"
        >
          PP-OCRv6 Tiny (RTen WASM)
        </span>
      </div>

      <div class="flex items-center gap-2">
        <span
          v-if="durationMs !== null"
          class="text-xs text-neutral-500 dark:text-neutral-400 font-mono"
        >
          {{ durationMs }} ms
        </span>
        <button
          type="button"
          :disabled="status !== 'ready' || isScanning"
          @click="runOcr"
          class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-primary text-white hover:opacity-90 disabled:opacity-50 transition-all cursor-pointer shadow-sm"
        >
          <span
            v-if="isScanning"
            class="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"
          />
          {{ isScanning ? 'Processing...' : 'Run OCR' }}
        </button>
      </div>
    </div>

    <!-- Quick Sample Selector -->
    <div
      class="p-4 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-100/50 dark:bg-neutral-900/50 flex flex-wrap items-center gap-3"
    >
      <span class="text-xs font-medium text-neutral-500 dark:text-neutral-400">Sample Image:</span>
      <button
        v-for="sample in samples"
        :key="sample.id"
        type="button"
        @click="pickSample(sample)"
        class="px-3 py-1 text-xs rounded-md border transition-all cursor-pointer"
        :class="
          selectedSample === sample.id
            ? 'border-primary bg-primary/10 text-primary font-medium'
            : 'border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 text-neutral-700 dark:text-neutral-300'
        "
      >
        {{ sample.name }}
      </button>

      <label
        class="px-3 py-1 text-xs rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 hover:border-primary text-neutral-600 dark:text-neutral-400 hover:text-primary transition-all cursor-pointer inline-flex items-center gap-1.5"
      >
        <span>+ Upload File</span>
        <input type="file" accept="image/*" class="hidden" @change="handleFileUpload" />
      </label>
    </div>

    <!-- Main Workspace: Image Visualizer + Inspector Panel -->
    <div class="grid grid-cols-1 lg:grid-cols-12 min-h-[480px]">
      <!-- Left: Interactive Image Canvas & Overlays (7 cols) -->
      <div
        class="lg:col-span-7 p-4 flex flex-col items-center justify-center bg-neutral-950/5 dark:bg-black/30 border-b lg:border-b-0 lg:border-r border-neutral-200 dark:border-neutral-800 overflow-auto"
      >
        <div
          class="relative max-w-full inline-block rounded-lg shadow-md overflow-hidden bg-white dark:bg-neutral-950"
        >
          <img
            ref="imgRef"
            :src="imageSrc"
            alt="OCR Target Document"
            @load="onImageLoad"
            class="block max-w-full max-h-[500px] w-auto h-auto object-contain"
          />

          <!-- Scanning Laser Bar Effect -->
          <div
            v-if="isScanning"
            class="absolute inset-0 pointer-events-none overflow-hidden bg-primary/5"
          >
            <div class="scanner-laser" />
          </div>

          <!-- Bounding Boxes Overlay -->
          <div
            v-if="!isScanning && results.length > 0"
            class="absolute inset-0 pointer-events-none"
          >
            <div
              v-for="(item, idx) in results"
              :key="idx"
              class="absolute pointer-events-auto transition-all cursor-pointer"
              :style="{
                left: `${item.frame.left * scaleX}px`,
                top: `${item.frame.top * scaleY}px`,
                width: `${item.frame.width * scaleX}px`,
                height: `${item.frame.height * scaleY}px`,
                border:
                  hoveredIndex === idx
                    ? '2px solid #f97316'
                    : '1.5px solid rgba(59, 130, 246, 0.7)',
                backgroundColor:
                  hoveredIndex === idx ? 'rgba(249, 115, 22, 0.25)' : 'rgba(59, 130, 246, 0.15)',
              }"
              @mouseenter="hoveredIndex = idx"
              @mouseleave="hoveredIndex = null"
              @click="copyToClipboard(item.text)"
              :title="`${item.text} (${Math.round(item.score * 100)}%) - Click to copy`"
            />
          </div>
        </div>

        <p class="mt-3 text-[11px] text-neutral-400">
          Tip: Hover over bounding boxes to highlight recognized text. Click to copy.
        </p>
      </div>

      <!-- Right: Inspector & Output Panel (5 cols) -->
      <div class="lg:col-span-5 flex flex-col bg-white dark:bg-neutral-900">
        <!-- Granularity & Tab Header -->
        <div
          class="p-3 border-b border-neutral-200 dark:border-neutral-800 flex items-center justify-between gap-2"
        >
          <div class="flex items-center gap-1">
            <button
              v-for="mode in ['lines', 'words', 'spatial'] as OutputGranularity[]"
              :key="mode"
              type="button"
              @click="setGranularity(mode)"
              class="px-2.5 py-1 text-[11px] rounded capitalize transition-all cursor-pointer font-medium"
              :class="
                outputMode === mode
                  ? 'bg-primary text-white'
                  : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800'
              "
            >
              {{ mode }}
            </button>
          </div>

          <button
            type="button"
            @click="
              copyToClipboard(
                activeTab === 'spatial'
                  ? spatialText
                  : activeTab === 'csv'
                    ? csvExport
                    : activeTab === 'json'
                      ? jsonExport
                      : results.map((r) => r.text).join('\n')
              )
            "
            class="px-2.5 py-1 text-[11px] rounded border border-neutral-200 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-all cursor-pointer"
          >
            {{ copied ? '✓ Copied' : 'Copy All' }}
          </button>
        </div>

        <!-- Secondary Tabs: Structured List / Spatial / CSV / JSON -->
        <div
          class="flex border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50 text-[11px]"
        >
          <button
            type="button"
            @click="activeTab = 'structured'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer"
            :class="
              activeTab === 'structured'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            Items ({{ isScanning ? '...' : results.length }})
          </button>
          <button
            type="button"
            @click="activeTab = 'spatial'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer"
            :class="
              activeTab === 'spatial'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            Spatial Layout
          </button>
          <button
            type="button"
            @click="activeTab = 'csv'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer"
            :class="
              activeTab === 'csv'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            CSV
          </button>
          <button
            type="button"
            @click="activeTab = 'json'"
            class="px-3 py-1.5 border-b-2 font-medium transition-all cursor-pointer"
            :class="
              activeTab === 'json'
                ? 'border-primary text-primary'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
            "
          >
            JSON
          </button>
        </div>

        <!-- Tab Body Content -->
        <div class="flex-1 p-3 overflow-y-auto max-h-[420px] text-xs font-mono">
          <!-- Scanning Skeleton State -->
          <template v-if="isScanning">
            <!-- Structured items skeleton -->
            <div v-if="activeTab === 'structured'" class="space-y-2">
              <div
                v-for="i in 5"
                :key="i"
                class="p-2.5 rounded border border-neutral-200/70 dark:border-neutral-800/70 bg-neutral-100/60 dark:bg-neutral-900/60 flex items-center justify-between gap-3 animate-pulse"
              >
                <div
                  class="h-3.5 rounded bg-neutral-200 dark:bg-neutral-800"
                  :style="{ width: `${40 + ((i * 13) % 45)}%` }"
                />
                <div class="h-4 w-9 rounded bg-neutral-200 dark:bg-neutral-800 shrink-0" />
              </div>
            </div>

            <!-- Spatial ASCII layout skeleton -->
            <div
              v-else-if="activeTab === 'spatial'"
              class="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 space-y-3 animate-pulse"
            >
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-2/5" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-3/5 ml-12" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-4/5 ml-4" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-1/2" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-3/4 ml-8" />
              <div class="h-3 rounded bg-neutral-200 dark:bg-neutral-800 w-2/3 ml-16" />
            </div>

            <!-- Raw Tabular / JSON / CSV skeleton -->
            <div
              v-else
              class="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 space-y-2.5 animate-pulse"
            >
              <div
                v-for="i in 6"
                :key="i"
                class="h-3 rounded bg-neutral-200 dark:bg-neutral-800"
                :style="{ width: `${30 + ((i * 17) % 65)}%` }"
              />
            </div>
          </template>

          <!-- Resolved OCR Output -->
          <template v-else>
            <!-- Structured list view -->
            <div v-if="activeTab === 'structured'" class="space-y-1.5">
              <div
                v-for="(item, idx) in results"
                :key="idx"
                @mouseenter="hoveredIndex = idx"
                @mouseleave="hoveredIndex = null"
                class="p-2 rounded border transition-all flex items-start justify-between gap-2"
                :class="
                  hoveredIndex === idx
                    ? 'border-primary bg-primary/5 dark:bg-primary/10'
                    : 'border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-900/50'
                "
              >
                <div class="flex-1 font-sans text-neutral-800 dark:text-neutral-200 select-all">
                  {{ item.text }}
                </div>
                <span
                  class="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold"
                  :class="
                    item.score >= 0.9
                      ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-400'
                      : 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-400'
                  "
                >
                  {{ Math.round(item.score * 100) }}%
                </span>
              </div>
              <p v-if="results.length === 0" class="text-neutral-400 p-4 text-center">
                No text detected or score below threshold.
              </p>
            </div>

            <!-- Spatial ASCII view -->
            <pre
              v-else-if="activeTab === 'spatial'"
              class="whitespace-pre overflow-x-auto text-neutral-800 dark:text-neutral-200 p-3 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 font-mono text-xs leading-relaxed select-text shadow-inner"
              style="
                font-family:
                  ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono',
                  'Courier New', monospace;
                tab-size: 4;
              "
              >{{
                spatialText || 'Run detection to see reconstructed 2D spatial text layout.'
              }}</pre>

            <!-- CSV view -->
            <pre
              v-else-if="activeTab === 'csv'"
              class="whitespace-pre overflow-x-auto text-neutral-700 dark:text-neutral-300 p-3 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 font-mono text-xs leading-relaxed select-text"
              style="
                font-family:
                  ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono',
                  'Courier New', monospace;
                tab-size: 4;
              "
              >{{ csvExport }}</pre>

            <!-- JSON view -->
            <pre
              v-else-if="activeTab === 'json'"
              class="whitespace-pre overflow-x-auto text-neutral-700 dark:text-neutral-300 p-3 rounded-lg bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 font-mono text-xs leading-relaxed select-text"
              style="
                font-family:
                  ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono',
                  'Courier New', monospace;
                tab-size: 4;
              "
              >{{ jsonExport }}</pre>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.rusto-demo-container {
  font-family: inherit;
}

@keyframes scanline {
  0% {
    top: 0%;
  }
  50% {
    top: 96%;
  }
  100% {
    top: 0%;
  }
}

.scanner-laser {
  position: absolute;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent 0%, rgba(59, 130, 246, 0.9) 50%, transparent 100%);
  box-shadow: 0 0 12px 2px rgba(59, 130, 246, 0.7);
  animation: scanline 2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}
</style>
