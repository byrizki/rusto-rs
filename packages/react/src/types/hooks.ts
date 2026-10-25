import type {
  DetectTextOptions,
  ImageSource,
  TextResult,
} from '@rustors/web';

/**
 * Options for configuring the `useOcr` hook.
 */
export interface UseOcrOptions extends DetectTextOptions {
  /** Callback fired when text detection completes successfully. */
  onSuccess?: (results: TextResult[] | string) => void;
  /** Callback fired when an error occurs during OCR detection. */
  onError?: (error: Error) => void;
}

/**
 * Return value of the `useOcr` hook containing state and actions.
 */
export interface UseOcrReturn {
  /** Trigger OCR detection on an image source. */
  detect: (
    source: ImageSource,
    overrideOptions?: DetectTextOptions
  ) => Promise<TextResult[] | string>;
  /** Structured OCR results (empty when output is 'spatial'). */
  results: TextResult[];
  /** Reconstructed spatial text layout (empty when output is 'lines' or 'words'). */
  spatialText: string;
  /** Whether OCR detection is currently in progress. */
  isProcessing: boolean;
  /** Error object if OCR detection failed. */
  error: Error | null;
  /** Reset results, spatialText, isProcessing, and error back to initial states. */
  reset: () => void;
}
