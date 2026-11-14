<script setup lang="ts">
import { computed } from 'vue';
import OcrInfoPopover from './OcrInfoPopover.vue';

const props = withDefaults(
  defineProps<{
    settingId?: string;
    modelValue: number;
    label: string;
    identifier?: string;
    tooltip?: string;
    min: number;
    max: number;
    step: number;
    unit?: string;
    decimals?: number;
    defaultValue?: number;
    disabled?: boolean;
    disabledReason?: string;
  }>(),
  {
    settingId: undefined,
    identifier: undefined,
    tooltip: undefined,
    unit: '',
    decimals: 2,
    defaultValue: undefined,
    disabled: false,
    disabledReason: undefined,
  }
);

const emit = defineEmits<{
  'update:modelValue': [val: number];
}>();

const isModified = computed<boolean>(
  () => props.defaultValue !== undefined && props.modelValue !== props.defaultValue
);

function clamp(val: number): number {
  const clamped = Math.min(props.max, Math.max(props.min, val));
  const factor = 10 ** props.decimals;
  return Math.round(clamped * factor) / factor;
}

function onInput(event: Event) {
  const raw = Number((event.target as HTMLInputElement).value);
  if (!Number.isNaN(raw)) {
    emit('update:modelValue', clamp(raw));
  }
}

function onReset() {
  if (props.defaultValue !== undefined) {
    emit('update:modelValue', props.defaultValue);
  }
}
</script>

<template>
  <div
    :data-setting-id="settingId"
    class="space-y-1.5 py-1.5 transition-opacity"
    :class="{ 'opacity-45 pointer-events-none': disabled }"
  >
    <div class="flex items-center justify-between gap-2">
      <div class="flex items-center gap-1.5 min-w-0 flex-wrap">
        <span
          v-if="isModified"
          class="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0"
          title="Modified from default"
          aria-label="Modified from default"
        />
        <label class="font-medium text-zinc-700 dark:text-zinc-300 text-xs">
          {{ label }}
        </label>
        <code
          v-if="identifier"
          class="text-[10px] font-mono text-zinc-400 dark:text-zinc-500 bg-zinc-100/80 dark:bg-zinc-800/70 px-1 py-0.5 rounded"
        >
          {{ identifier }}
        </code>
        <OcrInfoPopover v-if="tooltip" :text="tooltip" :code="identifier" />
      </div>

      <div class="flex items-center gap-1.5 shrink-0">
        <button
          v-if="isModified"
          type="button"
          @click="onReset"
          :title="`Reset to ${defaultValue}${unit}`"
          :aria-label="`Reset ${label} to default`"
          class="p-0.5 rounded text-zinc-400 hover:text-primary transition-colors cursor-pointer"
        >
          <svg
            class="w-3 h-3"
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
        <div class="relative flex items-center">
          <input
            :value="modelValue"
            type="number"
            :min="min"
            :max="max"
            :step="step"
            :disabled="disabled"
            @change="onInput"
            class="w-18 px-1.5 py-0.5 text-[11px] font-mono font-semibold text-right rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-zinc-800 dark:text-zinc-200 tabular-nums focus:outline-none focus:border-primary"
          />
          <span v-if="unit" class="ml-1 text-[10px] font-mono text-zinc-400">{{ unit }}</span>
        </div>
      </div>
    </div>

    <div class="flex items-center gap-2">
      <span class="text-[10px] font-mono text-zinc-400 tabular-nums w-8 text-right shrink-0">
        {{ min }}
      </span>
      <input
        :value="modelValue"
        type="range"
        :min="min"
        :max="max"
        :step="step"
        :disabled="disabled"
        @input="onInput"
        class="w-full accent-primary cursor-pointer disabled:cursor-not-allowed"
      />
      <span class="text-[10px] font-mono text-zinc-400 tabular-nums w-8 shrink-0">
        {{ max }}
      </span>
    </div>

    <p
      v-if="disabled && disabledReason"
      class="text-[10px] font-mono text-amber-600 dark:text-amber-400"
    >
      {{ disabledReason }}
    </p>
  </div>
</template>
