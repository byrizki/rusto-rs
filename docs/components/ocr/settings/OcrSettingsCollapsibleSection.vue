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
  'update:modelValue': [val: boolean];
}>();

function toggle() {
  emit('update:modelValue', !props.modelValue);
}
</script>

<template>
  <div
    class="rounded-xl border transition-all duration-200 overflow-hidden"
    :class="
      modelValue
        ? 'border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 shadow-2xs'
        : 'border-zinc-200/80 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50 hover:border-zinc-300 dark:hover:border-zinc-700'
    "
  >
    <button
      type="button"
      @click="toggle"
      class="w-full px-3 py-2.5 flex items-center justify-between gap-2.5 text-left cursor-pointer transition-colors"
      :aria-expanded="modelValue"
    >
      <div class="min-w-0 flex-1 flex items-center gap-2">
        <span
          v-if="activeCount > 0"
          class="w-2 h-2 rounded-full bg-emerald-500 shrink-0"
          :title="`${activeCount} customized`"
          :aria-label="`${activeCount} customized`"
        />
        <span class="font-semibold text-zinc-900 dark:text-zinc-100 text-[11px] truncate">
          {{ title }}
        </span>
        <OcrInfoPopover v-if="description || badge" :text="description ?? ''" :code="badge" />
      </div>

      <svg
        class="w-3.5 h-3.5 shrink-0 text-zinc-400 transition-transform duration-200"
        :class="{ 'rotate-180': modelValue }"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        stroke-width="2"
        aria-hidden="true"
      >
        <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
      </svg>
    </button>

    <div
      v-if="modelValue"
      class="px-3 pb-3 pt-1 border-t border-zinc-100 dark:border-zinc-800/80 space-y-3"
    >
      <slot />
    </div>
  </div>
</template>
