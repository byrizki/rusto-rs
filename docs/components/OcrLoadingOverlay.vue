<script setup lang="ts">
import { computed } from 'vue';
import type { ProgressState } from '../types/ocrConfig';

const props = defineProps<{
  progress: ProgressState;
}>();

const radius = 26;
const circumference = 2 * Math.PI * radius;

const shouldDisplay = computed(
  () =>
    props.progress.active &&
    (props.progress.phase === 'download' || props.progress.phase === 'init')
);
</script>

<template>
  <Transition
    enter-active-class="transition duration-200 ease-out"
    enter-from-class="opacity-0 scale-98"
    enter-to-class="opacity-100 scale-100"
    leave-active-class="transition duration-150 ease-in"
    leave-from-class="opacity-100 scale-100"
    leave-to-class="opacity-0 scale-98"
  >
    <div
      v-if="shouldDisplay"
      class="absolute inset-0 z-40 flex flex-col items-center justify-center bg-white/80 dark:bg-zinc-950/80 backdrop-blur-sm select-none p-4 cursor-wait"
      role="status"
      aria-live="polite"
      @click.stop
    >
      <div
        class="flex flex-col items-center justify-center p-6 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-xl max-w-xs w-full mx-4 text-center"
      >
        <div class="relative w-16 h-16 flex items-center justify-center my-1.5">
          <svg class="w-full h-full -rotate-90" viewBox="0 0 64 64" aria-hidden="true">
            <circle
              cx="32"
              cy="32"
              :r="radius"
              fill="none"
              stroke="currentColor"
              stroke-width="4.5"
              class="text-zinc-200 dark:text-zinc-800"
            />
            <circle
              cx="32"
              cy="32"
              :r="radius"
              fill="none"
              stroke="currentColor"
              stroke-width="4.5"
              stroke-linecap="round"
              class="text-primary transition-[stroke-dashoffset] duration-300 ease-out"
              :stroke-dasharray="circumference"
              :stroke-dashoffset="
                circumference - (circumference * Math.min(Math.max(progress.percent, 0), 100)) / 100
              "
            />
          </svg>

          <div class="absolute inset-0 flex items-center justify-center">
            <span
              v-if="progress.percent > 0"
              class="text-xs font-bold font-mono text-zinc-800 dark:text-zinc-100 tabular-nums"
            >
              {{ progress.percent }}%
            </span>
            <span
              v-else
              class="inline-block w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"
            />
          </div>
        </div>

        <div class="mt-2 flex items-center justify-center gap-1.5 flex-wrap">
          <span
            v-if="progress.phase === 'download'"
            class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-sky-50 dark:bg-sky-950/60 text-sky-700 dark:text-sky-300 border border-sky-200/80 dark:border-sky-900/70"
          >
            <span class="w-1.5 h-1.5 rounded-full bg-sky-500 animate-pulse" />
            Downloading Weights
          </span>
          <span
            v-else
            class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 border border-amber-200/80 dark:border-amber-900/70"
          >
            <span class="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            Initializing WASM
          </span>
        </div>

        <h4 class="text-xs font-semibold text-zinc-900 dark:text-zinc-100 mt-2.5">
          {{ progress.stageText || 'Preparing engine...' }}
        </h4>

        <p
          v-if="progress.detailText"
          class="text-[11px] text-zinc-500 dark:text-zinc-400 font-mono mt-1 tabular-nums"
        >
          {{ progress.detailText }}
        </p>
      </div>
    </div>
  </Transition>
</template>
