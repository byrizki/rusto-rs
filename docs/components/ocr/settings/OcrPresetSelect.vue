<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue';
import type { OcrProfilePreset, PresetCategory } from '../../../types/ocrPresets';

const props = defineProps<{
  presets: OcrProfilePreset[];
  activePreset: OcrProfilePreset | null;
  isConfigModified?: boolean;
}>();

const emit = defineEmits<{
  select: [preset: OcrProfilePreset];
}>();

const isOpen = ref(false);
const searchQuery = ref('');
const rootRef = ref<HTMLElement | null>(null);
const searchInputRef = ref<HTMLInputElement | null>(null);

const categories: PresetCategory[] = [
  'Identity & Finance',
  'Documents & Publishing',
  'Performance & Camera',
];

const filteredPresets = computed(() => {
  const q = searchQuery.value.trim().toLowerCase();
  if (!q) return props.presets;

  return props.presets.filter((preset) => {
    const matchName = preset.name.toLowerCase().includes(q);
    const matchCat = preset.category.toLowerCase().includes(q);
    const matchDesc = preset.description.toLowerCase().includes(q);
    const matchKeywords = preset.keywords?.some((k) => k.toLowerCase().includes(q)) ?? false;
    return matchName || matchCat || matchDesc || matchKeywords;
  });
});

const groupedPresets = computed(() => {
  const groups: Array<{ category: PresetCategory; presets: OcrProfilePreset[] }> = [];
  for (const cat of categories) {
    const inCat = filteredPresets.value.filter((p) => p.category === cat);
    if (inCat.length > 0) {
      groups.push({ category: cat, presets: inCat });
    }
  }
  return groups;
});

function toggleDropdown() {
  isOpen.value = !isOpen.value;
  if (isOpen.value) {
    nextTick(() => {
      searchInputRef.value?.focus();
    });
  } else {
    searchQuery.value = '';
  }
}

function selectPreset(preset: OcrProfilePreset) {
  emit('select', preset);
  isOpen.value = false;
  searchQuery.value = '';
}

function handleKeyDown(e: KeyboardEvent) {
  if (e.key === 'Escape' && isOpen.value) {
    isOpen.value = false;
    searchQuery.value = '';
  }
}

function handleClickOutside(e: MouseEvent) {
  if (rootRef.value && !rootRef.value.contains(e.target as Node)) {
    isOpen.value = false;
    searchQuery.value = '';
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside);
  document.addEventListener('keydown', handleKeyDown);
});

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside);
  document.removeEventListener('keydown', handleKeyDown);
});
</script>

<template>
  <div ref="rootRef" class="relative w-full text-xs">
    <!-- Compact Combobox Trigger -->
    <div
      class="flex items-center justify-between gap-2 px-3 py-1.5 rounded-xl border transition-all cursor-pointer select-none"
      :class="[
        isOpen
          ? 'border-primary bg-white dark:bg-zinc-900'
          : 'border-zinc-200 dark:border-zinc-800 bg-white/90 dark:bg-zinc-900/90 hover:border-zinc-300 dark:hover:border-zinc-700',
      ]"
      @click="toggleDropdown"
      role="combobox"
      :aria-expanded="isOpen"
      aria-haspopup="listbox"
    >
      <div class="flex items-center gap-2 min-w-0">
        <span class="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 shrink-0">
          Preset:
        </span>
        <div class="flex items-center gap-1.5 truncate">
          <span
            class="font-semibold truncate"
            :class="
              activePreset
                ? 'text-zinc-900 dark:text-zinc-100'
                : 'text-amber-600 dark:text-amber-400'
            "
          >
            {{ activePreset?.name ?? 'Custom / Tuned' }}
          </span>
          <span
            v-if="activePreset?.badge"
            class="px-1.5 py-0.2 rounded text-[9px] font-medium bg-primary/10 text-primary shrink-0"
          >
            {{ activePreset.badge }}
          </span>
          <span
            v-else-if="!activePreset"
            class="px-1.5 py-0.2 rounded text-[9px] font-medium bg-amber-100 dark:bg-amber-950/80 text-amber-700 dark:text-amber-300 shrink-0"
          >
            Modified
          </span>
        </div>
      </div>

      <div class="flex items-center gap-1.5 text-zinc-400 shrink-0">
        <span class="text-[10px] hidden sm:inline text-zinc-500">
          {{ isOpen ? 'Close' : 'Change' }}
        </span>
        <svg
          class="w-3.5 h-3.5 transition-transform duration-200"
          :class="{ 'rotate-180': isOpen }"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          stroke-width="2"
        >
          <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </div>

    <!-- Complete Selected Preset Description -->
    <div
      v-if="activePreset?.description"
      class="mt-1.5 px-1 text-[11px] text-zinc-500 dark:text-zinc-400 leading-snug flex items-start gap-1.5"
    >
      <span class="text-primary mt-0.5 shrink-0 text-[10px]">ℹ</span>
      <p class="flex-1">{{ activePreset.description }}</p>
    </div>
    <div
      v-else-if="isConfigModified"
      class="mt-1.5 px-1 text-[11px] text-amber-600 dark:text-amber-400 leading-snug flex items-start gap-1.5"
    >
      <span class="mt-0.5 shrink-0 text-[10px]">●</span>
      <p class="flex-1">Settings have been customized from default preset values.</p>
    </div>

    <!-- Dropdown Menu / Searchable Selection -->
    <Transition
      enter-active-class="transition duration-100 ease-out"
      enter-from-class="opacity-0 scale-98 translate-y--1"
      enter-to-class="opacity-100 scale-100 translate-y-0"
      leave-active-class="transition duration-75 ease-in"
      leave-from-class="opacity-100 scale-100 translate-y-0"
      leave-to-class="opacity-0 scale-98 translate-y--1"
    >
      <div
        v-if="isOpen"
        class="absolute left-0 right-0 top-full mt-1.5 z-50 rounded-xl shadow-md border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 overflow-hidden text-xs flex flex-col max-h-[320px]"
        role="listbox"
      >
        <!-- Search Input Bar -->
        <div
          class="p-2 border-b border-zinc-100 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-950/50 shrink-0"
        >
          <div class="relative flex items-center">
            <svg
              class="w-3.5 h-3.5 text-zinc-400 absolute left-2.5 pointer-events-none"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="2"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              ref="searchInputRef"
              v-model="searchQuery"
              type="text"
              placeholder="Search presets (e.g. receipt, mobile, table, book, ktp)..."
              class="w-full pl-8 pr-7 py-1 text-xs rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 focus:outline-none focus:border-primary"
            />
            <button
              v-if="searchQuery"
              type="button"
              @click="searchQuery = ''"
              class="absolute right-2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 cursor-pointer"
            >
              ✕
            </button>
          </div>
        </div>

        <!-- Scrollable Preset List -->
        <div class="overflow-y-auto p-1.5 space-y-2 flex-1">
          <div
            v-if="filteredPresets.length === 0"
            class="py-4 text-center text-zinc-400 text-[11px]"
          >
            No presets found matching "{{ searchQuery }}"
          </div>

          <div v-for="group in groupedPresets" :key="group.category" class="space-y-1">
            <div
              class="px-2 pt-1 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-400"
            >
              {{ group.category }}
            </div>

            <div
              v-for="preset in group.presets"
              :key="preset.id"
              @click="selectPreset(preset)"
              class="px-2.5 py-1.5 rounded-lg border transition-all cursor-pointer flex items-start justify-between gap-2"
              :class="[
                activePreset?.id === preset.id
                  ? 'border-primary/80 bg-primary/5 dark:bg-primary/10 text-primary font-medium'
                  : 'border-transparent hover:bg-zinc-100/70 dark:hover:bg-zinc-800/60 text-zinc-700 dark:text-zinc-300',
              ]"
              role="option"
              :aria-selected="activePreset?.id === preset.id"
            >
              <div class="min-w-0 flex-1">
                <div class="flex items-center gap-1.5 flex-wrap">
                  <span class="font-semibold text-zinc-900 dark:text-zinc-100 text-[11px]">
                    {{ preset.name }}
                  </span>
                  <span
                    v-if="preset.badge"
                    class="px-1.5 py-0.2 rounded text-[9px] font-medium bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400"
                  >
                    {{ preset.badge }}
                  </span>
                </div>
                <p class="text-[10px] text-zinc-500 dark:text-zinc-400 line-clamp-1 mt-0.5">
                  {{ preset.description }}
                </p>
              </div>

              <div class="shrink-0 pt-0.5">
                <svg
                  v-if="activePreset?.id === preset.id"
                  class="w-4 h-4 text-primary"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  stroke-width="2.5"
                >
                  <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>
