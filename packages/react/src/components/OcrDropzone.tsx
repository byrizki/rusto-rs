import {
  useState,
  useRef,
  useCallback,
  type ReactElement,
  type DragEvent,
  type ChangeEvent,
} from 'react';
import type { OcrDropzoneProps } from '../types/index.js';
import { DropzonePrompt } from './DropzonePrompt.js';
import { DropzonePreview } from './DropzonePreview.js';
import { useOcr } from '../hooks/useOcr.js';

export function OcrDropzone({
  onFileSelect,
  onDetect,
  autoDetect = false,
  detectOptions,
  accept = 'image/png,image/jpeg,image/webp,image/bmp',
  className,
  style,
}: OcrDropzoneProps): ReactElement {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const { detect } = useOcr({
    ...detectOptions,
    onSuccess: (res) => onDetect?.(res),
  });

  const handleFile = useCallback(
    async (file: File) => {
      onFileSelect?.(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);

      if (autoDetect) {
        await detect(file, detectOptions);
      }
    },
    [autoDetect, detect, detectOptions, onFileSelect]
  );

  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFile(files[0]);
      }
    },
    [handleFile]
  );

  const onInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleFile(files[0]);
      }
    },
    [handleFile]
  );

  const clearSelection = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [previewUrl]);

  return (
    <div
      className={className}
      onClick={() => fileInputRef.current?.click()}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      style={{
        border: `2px dashed ${isDragOver ? '#3b82f6' : '#d1d5db'}`,
        borderRadius: '8px',
        backgroundColor: isDragOver ? '#eff6ff' : '#fafafa',
        cursor: 'pointer',
        transition: 'border-color 0.15s ease, background-color 0.15s ease',
        boxSizing: 'border-box',
        ...style,
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={onInputChange}
        style={{ display: 'none' }}
      />
      {previewUrl ? (
        <DropzonePreview previewUrl={previewUrl} onClear={clearSelection} />
      ) : (
        <DropzonePrompt />
      )}
    </div>
  );
}
