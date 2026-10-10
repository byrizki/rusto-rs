import type {
  DetectTextOptions,
  ImageSource,
  TextResult,
} from '@rustors/web';

export interface UseOcrOptions extends DetectTextOptions {
  onSuccess?: (results: TextResult[] | string) => void;
  onError?: (error: Error) => void;
}

export interface UseOcrReturn {
  detect: (
    source: ImageSource,
    overrideOptions?: DetectTextOptions
  ) => Promise<TextResult[] | string>;
  results: TextResult[];
  spatialText: string;
  isProcessing: boolean;
  error: Error | null;
  reset: () => void;
}
