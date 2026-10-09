<script setup lang="ts">
import type { ProgressState } from '../types/ocrConfig';

defineProps<{
  progress: ProgressState;
}>();

const radius = 26;
const circumference = 2 * Math.PI * radius;
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
      v-if="progress.active"
      class="absolute inset-0 z-50 flex flex-col items-center justify-center bg-white/75 dark:bg-neutral-950/80 backdrop-blur-md select-none p-4 cursor-wait"
      role="status"
      aria-live="polite"
      @click.stop
    >
      <div
        class="flex flex-col items-center justify-center p-6 rounded-2xl bg-white/95 dark:bg-neutral-900/95 border border-neutral-200 dark:border-neutral-800 shadow-2xl max-w-sm w-full mx-4 text-center ring-1 ring-black/5 dark:ring-white/5"
      >
        <!-- Circular Progress Ring -->
        <div class="relative w-18 h-18 flex items-center justify-center my-2">
          <svg class="w-full h-full -rotate-90" viewBox="0 0 64 64">
            <!-- Background circle track -->
            <circle
              cx="32"
              cy="32"
              :r="radius"
              fill="none"
              stroke="currentColor"
              stroke-width="4.5"
              class="text-neutral-200 dark:text-neutral-800"
            />
            <!-- Progress circle fill -->
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

          <!-- Center Percentage or Spinner -->
          <div class="absolute inset-0 flex items-center justify-center">
            <span
              v-if="progress.percent > 0"
              class="text-xs font-bold font-mono text-neutral-800 dark:text-neutral-100"
            >
              {{ progress.percent }}%
            </span>
            <span
              v-else
              class="inline-block w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"
            />
          </div>
        </div>

        <!-- Phase & Pipeline Step Badges -->
        <div class="mt-2 flex items-center justify-center gap-1.5 flex-wrap">
          <span
            v-if="progress.step"
            class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 border border-neutral-200 dark:border-neutral-700 font-mono"
          >
            Step {{ progress.step.current }}/{{ progress.step.total }}
          </span>
          <span
            v-if="progress.phase === 'download'"
            class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-blue-50 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-900"
          >
            <span class="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
            Downloading Model
          </span>
          <span
            v-else-if="progress.phase === 'init'"
            class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-900"
          >
            <span class="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            Initializing Pipeline
          </span>
          <span
            v-else
            class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-primary/10 text-primary border border-primary/20"
          >
            <span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            Running Inference
          </span>
        </div>

        <!-- Stage Text -->
        <h4
          class="text-sm font-semibold text-neutral-900 dark:text-neutral-100 mt-2.5 tracking-tight"
        >
          {{ progress.stageText || 'Processing...' }}
        </h4>

        <!-- Detail Text -->
        <p
          v-if="progress.detailText"
          class="text-xs text-neutral-500 dark:text-neutral-400 font-mono mt-1 break-all"
        >
          {{ progress.detailText }}
        </p>
      </div>
    </div>
  </Transition>
</template>
