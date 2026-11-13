<script setup lang="ts">
import { computed, onMounted, ref, watch, type CSSProperties } from 'vue';
import type { TextResult } from '@rustors/web';
import type { OcrDemoStrings } from '../../types/ocrDemoI18n';

const props = defineProps<{
  imageSrc: string;
  isScanning: boolean;
  results: TextResult[];
  hoveredIndex: number | null;
  hasRun: boolean;
  strings: OcrDemoStrings['canvas'];
}>();

const emit = defineEmits<{
  'update:hoveredIndex': [index: number | null];
  copyItem: [text: string, index: number];
  uploadFile: [file: File];
}>();

const imgRef = ref<HTMLImageElement | null>(null);
const naturalSize = ref({ width: 1, height: 1 });
const fittedWidth = ref<number>(0);
const showBoxes = ref<boolean>(true);
const zoom = ref<number>(1);
const isDragging = ref<boolean>(false);
const copiedToast = ref<string | null>(null);
let toastTimer: ReturnType<typeof setTimeout> | null = null;

watch(
  () => props.imageSrc,
  () => {
    zoom.value = 1;
    showBoxes.value = true;
  }
);

function syncNaturalDimensions() {
  const img = imgRef.value;
  if (!img) return;
  if (img.naturalWidth > 0 && img.naturalHeight > 0) {
    naturalSize.value = {
      width: img.naturalWidth,
      height: img.naturalHeight,
    };
  }
  const rect = img.getBoundingClientRect();
  if (zoom.value === 1 && rect.width > 0) {
    fittedWidth.value = rect.width;
  }
}

function adjustZoom(delta: number) {
  if (zoom.value === 1 && imgRef.value) {
    const rect = imgRef.value.getBoundingClientRect();
    if (rect.width > 0) fittedWidth.value = rect.width;
  }
  const next = Math.round((zoom.value + delta) * 100) / 100;
  zoom.value = Math.min(Math.max(next, 0.5), 2.5);
}

function resetZoom() {
  zoom.value = 1;
}

const zoomedStyle = computed<CSSProperties>(() => {
  if (zoom.value === 1 || fittedWidth.value <= 0) return {};
  return {
    width: `${Math.round(fittedWidth.value * zoom.value)}px`,
    maxWidth: 'none',
    maxHeight: 'none',
  };
});

function boxStyle(item: TextResult, idx: number) {
  const nw = naturalSize.value.width || 1;
  const nh = naturalSize.value.height || 1;
  const isHovered = props.hoveredIndex === idx;
  return {
    left: `${(item.frame.left / nw) * 100}%`,
    top: `${(item.frame.top / nh) * 100}%`,
    width: `${(item.frame.width / nw) * 100}%`,
    height: `${(item.frame.height / nh) * 100}%`,
    zIndex: isHovered ? 20 : 10,
  };
}

function onBoxClick(item: TextResult, idx: number) {
  emit('copyItem', item.text, idx);
  const preview = item.text.length > 32 ? `${item.text.slice(0, 32)}…` : item.text;
  copiedToast.value = preview;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    copiedToast.value = null;
  }, 1800);
}

function onDragOver(event: DragEvent) {
  event.preventDefault();
  isDragging.value = true;
}

function onDragLeave(event: DragEvent) {
  const currentTarget = event.currentTarget as HTMLElement | null;
  const related = event.relatedTarget as Node | null;
  if (currentTarget && related && currentTarget.contains(related)) return;
  isDragging.value = false;
}

function onDrop(event: DragEvent) {
  event.preventDefault();
  isDragging.value = false;
  const file = event.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    emit('uploadFile', file);
  }
}

const hasDimensions = computed(() => naturalSize.value.width > 1 && naturalSize.value.height > 1);

onMounted(() => {
  if (imgRef.value?.complete) {
    syncNaturalDimensions();
  }
});
</script>

<template>
  <section
    class="lg:col-span-7 relative flex flex-col bg-zinc-100/70 dark:bg-zinc-950/70 border-b lg:border-b-0 lg:border-r border-zinc-200/80 dark:border-zinc-800 h-full min-h-[340px] max-h-[520px] lg:max-h-full overflow-hidden canvas-grid-bg"
    @dragover="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDrop"
  >
    <div
      class="absolute top-2.5 right-2.5 z-30 inline-flex items-center gap-1 p-1 rounded-lg bg-white/90 dark:bg-zinc-900/90 backdrop-blur-md border border-zinc-200/80 dark:border-zinc-800 shadow-sm text-[11px]"
    >
      <button
        v-if="results.length > 0"
        type="button"
        @click="showBoxes = !showBoxes"
        class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md font-medium transition-colors cursor-pointer"
        :class="
          showBoxes
            ? 'bg-primary/10 text-primary'
            : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100'
        "
      >
        <svg
          class="w-3.5 h-3.5 shrink-0"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path v-if="showBoxes" d="M10 12.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
          <path
            v-if="showBoxes"
            fill-rule="evenodd"
            d="M.664 10.59a1.651 1.651 0 0 1 0-1.186A10.004 10.004 0 0 1 10 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0 1 10 17c-4.257 0-7.893-2.66-9.336-6.41ZM14 10a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z"
            clip-rule="evenodd"
          />
          <path
            v-else
            fill-rule="evenodd"
            d="M3.28 2.22a.75.75 0 0 0-1.06 1.06l14.5 14.5a.75.75 0 1 0 1.06-1.06l-1.745-1.745a10.029 10.029 0 0 0 3.3-4.38 1.651 1.651 0 0 0 0-1.185A10.004 10.004 0 0 0 9.999 3a9.956 9.956 0 0 0-4.744 1.194L3.28 2.22ZM7.752 6.69l1.092 1.092a2.5 2.5 0 0 1 3.374 3.373l1.091 1.092a4 4 0 0 0-5.557-5.557Z"
            clip-rule="evenodd"
          />
        </svg>
        <span>{{ showBoxes ? strings.boxesOn : strings.boxesOff }}</span>
      </button>

      <span v-if="results.length > 0" class="w-px h-3.5 bg-zinc-200 dark:bg-zinc-800 mx-0.5" />

      <button
        type="button"
        :disabled="zoom <= 0.5"
        @click="adjustZoom(-0.25)"
        class="px-1.5 py-0.5 rounded text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 disabled:opacity-40 cursor-pointer font-mono font-bold"
        aria-label="Zoom out"
      >
        −
      </button>
      <button
        type="button"
        @click="resetZoom"
        :title="strings.resetZoom"
        class="px-1.5 py-0.5 rounded text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 cursor-pointer font-mono tabular-nums min-w-[42px] text-center"
      >
        {{ Math.round(zoom * 100) }}%
      </button>
      <button
        type="button"
        :disabled="zoom >= 2.5"
        @click="adjustZoom(0.25)"
        class="px-1.5 py-0.5 rounded text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 disabled:opacity-40 cursor-pointer font-mono font-bold"
        aria-label="Zoom in"
      >
        +
      </button>
    </div>

    <div
      class="flex-1 min-h-0 p-4 sm:p-5 overflow-auto flex"
      :class="zoom === 1 ? 'items-center justify-center' : 'items-start justify-start'"
    >
      <div
        class="relative inline-block rounded-lg shadow-md ring-1 ring-black/10 dark:ring-white/10 bg-white dark:bg-zinc-900 select-none m-auto"
      >
        <img
          ref="imgRef"
          :src="imageSrc"
          alt="Selected document for OCR recognition"
          @load="syncNaturalDimensions"
          :style="zoomedStyle"
          class="block max-w-full max-h-[400px] lg:max-h-[560px] w-auto h-auto rounded-lg"
        />

        <div
          v-if="isScanning"
          class="absolute inset-0 pointer-events-none overflow-hidden rounded-lg bg-primary/5"
        >
          <div class="scanner-beam" />
        </div>

        <div
          v-if="!isScanning && showBoxes && results.length > 0"
          class="absolute inset-0 pointer-events-none"
        >
          <div
            v-for="(item, idx) in results"
            :key="idx"
            class="absolute pointer-events-auto rounded-[2px] transition-colors duration-150 cursor-pointer"
            :class="
              hoveredIndex === idx
                ? 'border-2 border-primary bg-primary/25 shadow-sm'
                : 'border border-sky-500/75 bg-sky-500/10 hover:border-primary hover:bg-primary/20'
            "
            :style="boxStyle(item, idx)"
            @mouseenter="emit('update:hoveredIndex', idx)"
            @mouseleave="emit('update:hoveredIndex', null)"
            @click="onBoxClick(item, idx)"
          >
            <span
              v-if="hoveredIndex === idx"
              class="pointer-events-none absolute -top-5 left-0 inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-zinc-900/95 dark:bg-zinc-100/95 text-white dark:text-zinc-900 text-[10px] font-mono font-semibold whitespace-nowrap shadow-md tabular-nums"
            >
              <span>#{{ idx + 1 }}</span>
              <span class="opacity-75">{{ Math.round(item.score * 100) }}%</span>
            </span>
          </div>
        </div>
      </div>
    </div>

    <Transition
      enter-active-class="transition duration-150 ease-out"
      enter-from-class="opacity-0 translate-y-1"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition duration-150 ease-in"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 translate-y-1"
    >
      <div
        v-if="copiedToast"
        class="pointer-events-none absolute bottom-10 left-1/2 -translate-x-1/2 z-30 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 text-xs font-medium shadow-lg max-w-[85%]"
      >
        <svg
          class="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-600 shrink-0"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fill-rule="evenodd"
            d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z"
            clip-rule="evenodd"
          />
        </svg>
        <span class="truncate">{{ strings.copiedItem }}: "{{ copiedToast }}"</span>
      </div>
    </Transition>

    <Transition
      enter-active-class="transition duration-150 ease-out"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition duration-100 ease-in"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div
        v-if="isDragging"
        class="pointer-events-none absolute inset-2 z-40 rounded-xl border-2 border-dashed border-primary bg-white/90 dark:bg-zinc-950/90 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center"
      >
        <div
          class="w-10 h-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center mb-2"
        >
          <svg class="w-5 h-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              d="M9.25 13.25a.75.75 0 0 0 1.5 0V4.636l2.955 3.129a.75.75 0 0 0 1.09-1.03l-4.25-4.5a.75.75 0 0 0-1.09 0l-4.25 4.5a.75.75 0 1 0 1.09 1.03L9.25 4.636v8.614Z"
            />
            <path
              d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z"
            />
          </svg>
        </div>
        <p class="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
          {{ strings.dropOverlayTitle }}
        </p>
        <p class="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
          {{ strings.dropOverlaySub }}
        </p>
      </div>
    </Transition>

    <footer
      class="px-3.5 py-1.5 border-t border-zinc-200/70 dark:border-zinc-800/80 bg-white/60 dark:bg-zinc-900/60 flex items-center justify-between gap-2 text-[11px] text-zinc-500 dark:text-zinc-400 shrink-0"
    >
      <span v-if="hasDimensions" class="font-mono tabular-nums shrink-0">
        {{ naturalSize.width }} × {{ naturalSize.height }} px
      </span>
      <span class="truncate ml-auto">
        {{ hasRun && results.length > 0 ? strings.hintActive : strings.hintIdle }}
      </span>
    </footer>
  </section>
</template>

<style scoped>
.canvas-grid-bg {
  background-image: radial-gradient(rgba(113, 113, 122, 0.16) 1px, transparent 1px);
  background-size: 16px 16px;
}

@keyframes scanSweep {
  0% {
    transform: translateY(0%);
    opacity: 0.9;
  }
  50% {
    transform: translateY(2400%);
    opacity: 1;
  }
  100% {
    transform: translateY(0%);
    opacity: 0.9;
  }
}

.scanner-beam {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4%;
  border-bottom: 2px solid var(--ui-primary, #f97316);
  background: linear-gradient(180deg, transparent 0%, rgba(249, 115, 22, 0.18) 100%);
  animation: scanSweep 2.1s cubic-bezier(0.4, 0, 0.2, 1) infinite;
  will-change: transform;
}
</style>
