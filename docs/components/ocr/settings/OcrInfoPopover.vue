<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref } from 'vue';

const props = withDefaults(
  defineProps<{
    text?: string;
    code?: string;
  }>(),
  {
    text: '',
    code: undefined,
  }
);

const isVisible = ref(false);
const triggerRef = ref<HTMLElement | null>(null);
const popoverRef = ref<HTMLElement | null>(null);
const pos = ref({ top: 0, left: 0, width: 260, placeBelow: false });
let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

function updatePosition() {
  if (!triggerRef.value) return;
  const rect = triggerRef.value.getBoundingClientRect();
  const width = Math.min(260, window.innerWidth - 24);

  // If there's enough room above (at least 120px), place above; otherwise below
  const placeBelow = rect.top < 120;
  const top = placeBelow ? rect.bottom + 8 : rect.top - 8;

  // Center horizontally relative to trigger, clamped to viewport margins
  let left = rect.left + rect.width / 2 - width / 2;
  left = Math.max(12, Math.min(left, window.innerWidth - width - 12));

  pos.value = { top, left, width, placeBelow };
}

function show() {
  if (hoverTimeout) clearTimeout(hoverTimeout);
  updatePosition();
  isVisible.value = true;
  nextTick(() => {
    updatePosition();
  });
}

function hide() {
  hoverTimeout = setTimeout(() => {
    isVisible.value = false;
  }, 120);
}

function toggle(e?: Event) {
  if (e) {
    e.preventDefault();
    e.stopPropagation();
  }
  if (hoverTimeout) clearTimeout(hoverTimeout);
  if (isVisible.value) {
    isVisible.value = false;
  } else {
    show();
  }
}

function handleClickOutside(e: MouseEvent) {
  const target = e.target as Node;
  if (
    triggerRef.value &&
    !triggerRef.value.contains(target) &&
    popoverRef.value &&
    !popoverRef.value.contains(target)
  ) {
    isVisible.value = false;
  }
}

function handleKeyDown(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    isVisible.value = false;
  }
}

function handleScrollOrResize() {
  if (isVisible.value) {
    updatePosition();
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside, true);
  document.addEventListener('keydown', handleKeyDown);
  window.addEventListener('scroll', handleScrollOrResize, true);
  window.addEventListener('resize', handleScrollOrResize);
});

onUnmounted(() => {
  if (hoverTimeout) clearTimeout(hoverTimeout);
  document.removeEventListener('click', handleClickOutside, true);
  document.removeEventListener('keydown', handleKeyDown);
  window.removeEventListener('scroll', handleScrollOrResize, true);
  window.removeEventListener('resize', handleScrollOrResize);
});
</script>

<template>
  <span
    ref="triggerRef"
    role="button"
    tabindex="0"
    @mouseenter="show"
    @mouseleave="hide"
    @click.prevent.stop="toggle"
    @keydown.enter.prevent.stop="toggle"
    @keydown.space.prevent.stop="toggle"
    class="relative inline-flex items-center justify-center w-3.5 h-3.5 rounded-full text-zinc-400 hover:text-primary hover:bg-primary/10 dark:hover:bg-primary/20 transition-colors cursor-pointer select-none focus:outline-none"
    :aria-label="text || 'More information'"
    :aria-expanded="isVisible"
  >
    <svg
      class="w-3 h-3 pointer-events-none"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2.2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>

    <!-- Teleport to body to avoid parent overflow:hidden / overflow-y:auto clipping -->
    <Teleport to="body">
      <Transition
        enter-active-class="transition duration-120 ease-out"
        enter-from-class="opacity-0 scale-95"
        enter-to-class="opacity-100 scale-100"
        leave-active-class="transition duration-90 ease-in"
        leave-from-class="opacity-100 scale-100"
        leave-to-class="opacity-0 scale-95"
      >
        <div
          v-if="isVisible"
          ref="popoverRef"
          @mouseenter="show"
          @mouseleave="hide"
          :style="{
            position: 'fixed',
            top: `${pos.top}px`,
            left: `${pos.left}px`,
            width: `${pos.width}px`,
            transform: pos.placeBelow ? 'none' : 'translateY(-100%)',
          }"
          class="z-[9999] p-2.5 rounded-xl shadow-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-zinc-800 dark:text-zinc-100 text-[11px] leading-relaxed select-text"
          role="tooltip"
        >
          <div class="space-y-1">
            <p class="font-normal font-sans text-zinc-700 dark:text-zinc-200">
              <slot>{{ text }}</slot>
            </p>
            <div
              v-if="code"
              class="pt-1 border-t border-zinc-100 dark:border-zinc-700/80 font-mono text-[9px] text-primary"
            >
              {{ code }}
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </span>
</template>
