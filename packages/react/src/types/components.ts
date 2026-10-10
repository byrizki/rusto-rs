import type { CSSProperties, ReactNode } from 'react';
import type { TextResult, DetectTextOptions } from '@rustors/web';

export interface BoundingBoxProps {
  item: TextResult;
  scaleX: number;
  scaleY: number;
  highlightColor?: string;
  showTooltip?: boolean;
  className?: string;
  style?: CSSProperties;
  onClick?: (item: TextResult) => void;
}

export interface OcrOverlayProps {
  imageSrc: string;
  results: TextResult[];
  naturalWidth?: number;
  naturalHeight?: number;
  highlightColor?: string;
  showTooltips?: boolean;
  className?: string;
  style?: CSSProperties;
  onItemClick?: (item: TextResult) => void;
}

export interface DropzonePromptProps {
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
  className?: string;
}

export interface DropzonePreviewProps {
  previewUrl: string;
  onClear: () => void;
  className?: string;
}

export interface OcrDropzoneProps {
  onFileSelect?: (file: File) => void;
  onDetect?: (results: TextResult[] | string) => void;
  autoDetect?: boolean;
  detectOptions?: DetectTextOptions;
  accept?: string;
  className?: string;
  style?: CSSProperties;
}
