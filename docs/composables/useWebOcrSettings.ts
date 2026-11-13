import { computed, ref } from 'vue';
import { defaultOcrConfig, type OcrConfig } from '../types/ocrConfig';
import type { OcrProfilePreset } from '../types/ocrPresets';

export type OcrSettingsTab = 'detection' | 'recognition' | 'runtime';

function mergeWithRuntimeConfig(
  current: OcrConfig,
  presetConfig?: Partial<OcrConfig>,
  overrides?: Partial<OcrConfig>
): OcrConfig {
  return {
    ...defaultOcrConfig,
    ...presetConfig,
    ...overrides,
    inferenceBackend: current.inferenceBackend,
    useWebGpu: current.useWebGpu,
  };
}

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
    ocrConfig.value = mergeWithRuntimeConfig(ocrConfig.value);
  }

  function applyPreset(preset: OcrProfilePreset, overrides?: Partial<OcrConfig>) {
    ocrConfig.value = mergeWithRuntimeConfig(ocrConfig.value, preset.config, overrides);
  }

  function isPresetActive(preset: OcrProfilePreset, selectedModelId?: string): boolean {
    if (selectedModelId && preset.modelId !== selectedModelId) return false;
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
