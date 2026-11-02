import { ref } from 'vue';
import { defaultOcrConfig, type OcrConfig } from '../types/ocrConfig';
import type { OcrProfilePreset } from '../types/ocrPresets';

export function useWebOcrSettings() {
  const ocrConfig = ref<OcrConfig>({ ...defaultOcrConfig });
  const isConfigOpen = ref<boolean>(false);
  const configTab = ref<'detection' | 'layout' | 'resizing' | 'preprocess' | 'models'>('detection');

  function toggleConfigPopover() {
    isConfigOpen.value = !isConfigOpen.value;
  }

  function resetConfigDefaults() {
    ocrConfig.value = { ...defaultOcrConfig };
  }

  function applyPreset(preset: OcrProfilePreset) {
    ocrConfig.value = { ...ocrConfig.value, ...preset.config };
  }

  function isPresetActive(preset: OcrProfilePreset): boolean {
    return Object.entries(preset.config).every(
      ([k, v]) => (ocrConfig.value as Record<string, unknown>)[k] === v
    );
  }

  return {
    ocrConfig,
    isConfigOpen,
    configTab,
    toggleConfigPopover,
    resetConfigDefaults,
    applyPreset,
    isPresetActive,
  };
}
