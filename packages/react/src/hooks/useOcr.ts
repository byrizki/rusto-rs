import { useState, useCallback, useContext } from 'react';
import type {
  DetectTextOptions,
  ImageSource,
  TextResult,
} from 'rusto-web';
import { RustoContext } from '../context/RustoContext.js';
import { defaultOcrService } from '../services/ocrService.js';
import type { UseOcrOptions, UseOcrReturn } from '../types/index.js';

export function useOcr(options: UseOcrOptions = {}): UseOcrReturn {
  const context = useContext(RustoContext);
  const service = context?.service || defaultOcrService;

  const [results, setResults] = useState<TextResult[]>([]);
  const [spatialText, setSpatialText] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  const reset = useCallback(() => {
    setResults([]);
    setSpatialText('');
    setIsProcessing(false);
    setError(null);
  }, []);

  const detect = useCallback(
    async (
      source: ImageSource,
      overrideOptions?: DetectTextOptions
    ): Promise<TextResult[] | string> => {
      setIsProcessing(true);
      setError(null);

      const mergedOptions: DetectTextOptions = {
        ...options,
        ...overrideOptions,
      };

      try {
        const output = await service.detect(source, mergedOptions);

        if (typeof output === 'string') {
          setSpatialText(output);
          setResults([]);
        } else {
          setResults(output);
          setSpatialText('');
        }

        options.onSuccess?.(output);
        return output;
      } catch (err) {
        const errorObj = err instanceof Error ? err : new Error(String(err));
        setError(errorObj);
        options.onError?.(errorObj);
        throw errorObj;
      } finally {
        setIsProcessing(false);
      }
    },
    [options, service]
  );

  return {
    detect,
    results,
    spatialText,
    isProcessing,
    error,
    reset,
  };
}
