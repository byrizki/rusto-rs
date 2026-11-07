<script setup lang="ts">
import OcrInfoPopover from './OcrInfoPopover.vue';

const props = withDefaults(
  defineProps<{
    title: string;
    badge?: string;
    description?: string;
    modelValue?: boolean;
    activeCount?: number;
  }>(),
  {
    badge: undefined,
    description: undefined,
    modelValue: false,
    activeCount: 0,
  }
);

const emit = defineEmits<{
  'update:modelValue': [value: boolean];
}>();

function toggle() {
  emit('update:modelValue', !props.modelValue);
}
</script>

<template>
  <div
    class="rounded-xl border transition-all duration-200 overflow-hidden"
    :class="[
      modelValue
        ? 'border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900'
        : 'border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50 hover:border-zinc-300 dark:hover:border-zinc-700',
    ]"
  >
    <button
      type="button"
      @click="toggle"
      class="w-full px-3 py-2.5 flex items-center justify-between gap-2.5 text-left cursor-pointer transition-colors"
      :aria-expanded="modelValue"
    >
      <div class="min-w-0 flex-1">
        <div class="flex items-center gap-2 flex-wrap">
          <span class="font-semibold text-zinc-900 dark:text-zinc-100 text-[11px]">
            {{ title }}
          </span>
          <span
            v-if="badge"
            class="font-mono text-[9px] px-1.5 py-0.2 rounded bg-primary/10 text-primary font-medium"
          >
            {{ badge }}
          </span>
          <span
            v-if="activeCount > 0"
            class="px-1.5 py-0.2 rounded-full text-[9px] font-semibold bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800"
          >
            {{ activeCount }} active
          </span>
          <OcrInfoPopover v-if="description" :text="description" />
        </div>
      </div>

      <div class="flex items-center gap-1 shrink-0 text-zinc-400">
        <span class="text-[10px] font-medium text-zinc-500 hidden sm:inline">
          {{ modelValue ? 'Collapse' : 'Expand' }}
        </span>
        <svg
          class="w-3.5 h-3.5 transition-transform duration-200"
          :class="{ 'rotate-180': modelValue }"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="2"
        >
          <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </button>

    <div
      v-if="modelValue"
      class="px-3 pb-3 pt-1 border-t border-zinc-100 dark:border-zinc-800/80 space-y-3"
    >
      <slot />
    </div>
  </div>
</template>
