import { computed, ref } from 'vue';
import { defaultOcrConfig, type OcrConfig } from '../types/ocrConfig';
import type { OcrProfilePreset } from '../types/ocrPresets';

export type OcrSettingsTab = 'detection' | 'recognition' | 'runtime';

export function useWebOcrSettings() {
  const ocrConfig = ref<OcrConfig>({ ...defaultOcrConfig });
  const isConfigOpen = ref<boolean>(false);
  const configTab = ref<OcrSettingsTab>('detection');

  const isCustomConfig = computed<boolean>(
    () => JSON.stringify(ocrConfig.value) !== JSON.stringify(defaultOcrConfig)
  );

  function toggleConfigPopover() {
    isConfigOpen.value = !isConfigOpen.value;
  }

  function openConfigTab(tab: OcrSettingsTab) {
    configTab.value = tab;
    isConfigOpen.value = true;
  }

  function resetConfigDefaults() {
    ocrConfig.value = { ...defaultOcrConfig };
  }

  function applyPreset(preset: OcrProfilePreset) {
    ocrConfig.value = {
      ...defaultOcrConfig,
      ...preset.config,
      useWebGpu: ocrConfig.value.useWebGpu,
    };
  }

  function isPresetActive(preset: OcrProfilePreset): boolean {
    return Object.entries(preset.config).every(
      ([k, v]) => (ocrConfig.value as unknown as Record<string, unknown>)[k] === v
    );
  }

  return {
    ocrConfig,
    isConfigOpen,
    configTab,
    isCustomConfig,
    toggleConfigPopover,
    openConfigTab,
    resetConfigDefaults,
    applyPreset,
    isPresetActive,
  };
}
