<script setup lang="ts">
import { computed } from 'vue';
import OcrInfoPopover from './OcrInfoPopover.vue';

export interface SettingsSubTab {
  id: string;
  label: string;
  badge?: string;
  description?: string;
  activeCount?: number;
}

const props = defineProps<{
  tabs: SettingsSubTab[];
  modelValue: string;
}>();

const emit = defineEmits<{
  'update:modelValue': [id: string];
}>();

const activeTabMeta = computed<SettingsSubTab | undefined>(() =>
  props.tabs.find((t) => t.id === props.modelValue)
);

function onTabKeydown(event: KeyboardEvent, index: number) {
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  event.preventDefault();
  const dir = event.key === 'ArrowRight' ? 1 : -1;
  const nextIndex = (index + dir + props.tabs.length) % props.tabs.length;
  const nextTab = props.tabs[nextIndex];
  if (nextTab) {
    emit('update:modelValue', nextTab.id);
  }
}
</script>

<template>
  <div class="space-y-3">
    <div
      class="flex items-center gap-4 border-b border-zinc-200 dark:border-zinc-800 overflow-x-auto"
      role="tablist"
    >
      <button
        v-for="(tab, idx) in tabs"
        :key="tab.id"
        type="button"
        role="tab"
        :aria-selected="modelValue === tab.id"
        :tabindex="modelValue === tab.id ? 0 : -1"
        @click="emit('update:modelValue', tab.id)"
        @keydown="onTabKeydown($event, idx)"
        class="relative inline-flex items-center gap-1.5 pb-2 pt-0.5 text-[11px] font-medium border-b-2 -mb-px transition-colors cursor-pointer whitespace-nowrap focus-visible:outline-none focus-visible:text-primary"
        :class="
          modelValue === tab.id
            ? 'border-primary text-zinc-900 dark:text-zinc-100 font-semibold'
            : 'border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
        "
      >
        <span class="font-mono text-[10px] text-zinc-400">{{ idx + 1 }}.</span>
        <span>{{ tab.label }}</span>
        <span
          v-if="(tab.activeCount ?? 0) > 0"
          class="px-1 py-0 text-[9px] font-mono rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-300 tabular-nums"
          :title="`${tab.activeCount} modified`"
        >
          {{ tab.activeCount }}
        </span>
      </button>
    </div>

    <div
      v-if="activeTabMeta?.description || activeTabMeta?.badge"
      class="flex items-center justify-between gap-2 px-2 py-1.5 rounded-md bg-zinc-50/80 dark:bg-zinc-950/50 border border-zinc-200/60 dark:border-zinc-800/80"
    >
      <p class="text-[11px] text-zinc-500 dark:text-zinc-400 truncate">
        {{ activeTabMeta.description }}
      </p>
      <div class="flex items-center gap-1.5 shrink-0">
        <code
          v-if="activeTabMeta.badge"
          class="text-[10px] font-mono text-zinc-600 dark:text-zinc-300 bg-white dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800 px-1.5 py-0.5 rounded"
        >
          {{ activeTabMeta.badge }}
        </code>
        <OcrInfoPopover
          v-if="activeTabMeta.badge"
          :text="activeTabMeta.description ?? ''"
          :code="activeTabMeta.badge"
        />
      </div>
    </div>

    <div class="space-y-2.5">
      <slot />
    </div>
  </div>
</template>
