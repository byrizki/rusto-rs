<template>
  <div
    class="my-6 overflow-x-auto rounded-xl border border-gray-200/80 bg-gray-50/50 p-4 dark:border-gray-800 dark:bg-gray-900/50"
  >
    <div
      v-if="svgContent"
      class="mermaid-container flex justify-center [&_svg]:max-w-full [&_svg]:h-auto"
      v-html="svgContent"
    />
    <div v-else-if="error" class="text-xs text-red-500 font-mono p-2">
      Failed to render diagram: {{ error }}
      <pre class="mt-2 text-gray-400 text-[10px]">{{ code }}</pre>
    </div>
    <div v-else class="flex items-center justify-center py-6 text-sm text-gray-400">
      <span class="inline-block animate-pulse">Rendering diagram...</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';

const props = defineProps<{
  code: string;
}>();

const svgContent = ref<string>('');
const error = ref<string>('');
const uniqueId = `mermaid-${Math.random().toString(36).substring(2, 9)}`;

const renderDiagram = async () => {
  if (typeof window === 'undefined') return;
  try {
    error.value = '';
    const mermaidModule = await import('mermaid');
    const mermaid = mermaidModule.default || mermaidModule;

    const isDark = document.documentElement.classList.contains('dark');

    mermaid.initialize({
      startOnLoad: false,
      theme: isDark ? 'dark' : 'default',
      securityLevel: 'loose',
      fontFamily: 'Inter, system-ui, sans-serif',
      themeVariables: isDark
        ? {
            primaryColor: '#3b82f6',
            primaryTextColor: '#f8fafc',
            primaryBorderColor: '#60a5fa',
            lineColor: '#94a3b8',
            secondaryColor: '#1e293b',
            tertiaryColor: '#0f172a',
            background: '#0f172a',
            mainBkg: '#1e293b',
            nodeBorder: '#3b82f6',
          }
        : {
            primaryColor: '#e0e7ff',
            primaryTextColor: '#1e293b',
            primaryBorderColor: '#6366f1',
            lineColor: '#64748b',
            secondaryColor: '#f1f5f9',
            tertiaryColor: '#ffffff',
            background: '#ffffff',
            mainBkg: '#f8fafc',
            nodeBorder: '#6366f1',
          },
    });

    // Clean up code content: decode html entities if any
    let cleanCode = props.code.trim();
    cleanCode = cleanCode
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&amp;/g, '&')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/&le;/g, '<=')
      .replace(/&ge;/g, '>=')
      .replace(/&rarr;/g, '->')
      .replace(/&times;/g, 'x');

    const renderId = `mermaid-${Math.random().toString(36).substring(2, 9)}`;
    const { svg } = await mermaid.render(renderId, cleanCode);
    svgContent.value = svg;
  } catch (err: any) {
    console.error('Mermaid render error:', err);
    error.value = err?.message || 'Unknown error';
  }
};

onMounted(() => {
  renderDiagram();

  // Watch for theme changes (dark / light mode toggle)
  const observer = new MutationObserver(() => {
    renderDiagram();
  });

  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  });
});

watch(
  () => props.code,
  () => {
    renderDiagram();
  }
);
</script>
