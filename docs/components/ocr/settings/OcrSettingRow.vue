<script setup lang="ts">
import OcrInfoPopover from './OcrInfoPopover.vue';

withDefaults(
  defineProps<{
    settingId?: string;
    label: string;
    identifier?: string;
    description?: string;
    tooltip?: string;
    modelBadge?: string;
    modelHelp?: string;
    disabled?: boolean;
    disabledReason?: string;
    isModified?: boolean;
  }>(),
  {
    settingId: undefined,
    identifier: undefined,
    description: undefined,
    tooltip: undefined,
    modelBadge: undefined,
    modelHelp: undefined,
    disabled: false,
    disabledReason: undefined,
    isModified: false,
  }
);
</script>

<template>
  <div
    :data-setting-id="settingId"
    class="py-2 first:pt-0 last:pb-0 border-b border-zinc-100 dark:border-zinc-800/80 last:border-b-0 transition-opacity"
    :class="{ 'opacity-50': disabled }"
  >
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0 flex-1 space-y-0.5">
        <div class="flex items-center gap-1.5 flex-wrap">
          <span
            v-if="isModified"
            class="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0"
            title="Modified from default"
            aria-label="Modified from default"
          />
          <span class="font-medium text-zinc-800 dark:text-zinc-200 text-xs">
            {{ label }}
          </span>
          <code
            v-if="identifier"
            class="text-[10px] font-mono text-zinc-400 dark:text-zinc-500 bg-zinc-100/80 dark:bg-zinc-800/70 px-1 py-0.5 rounded"
          >
            {{ identifier }}
          </code>
          <span
            v-if="modelBadge"
            :title="modelHelp"
            class="px-1.5 py-0.5 text-[10px] font-mono font-medium rounded border border-amber-300/70 dark:border-amber-800/60 bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300"
          >
            {{ modelBadge }}
          </span>
          <OcrInfoPopover v-if="tooltip" :text="tooltip" :code="identifier" />
        </div>
        <p v-if="description" class="text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug">
          {{ description }}
        </p>
        <p
          v-if="modelHelp"
          class="text-[10px] font-mono text-zinc-500 dark:text-zinc-400 leading-snug"
        >
          {{ modelHelp }}
        </p>
        <p
          v-if="disabled && disabledReason"
          class="text-[10px] font-mono text-amber-600 dark:text-amber-400"
        >
          {{ disabledReason }}
        </p>
      </div>

      <div
        class="shrink-0 flex items-center gap-2 pt-0.5"
        :class="{ 'pointer-events-none': disabled }"
      >
        <slot />
      </div>
    </div>

    <div v-if="$slots.params" class="mt-2 pl-2 border-l-2 border-zinc-200 dark:border-zinc-800">
      <slot name="params" />
    </div>
  </div>
</template>
