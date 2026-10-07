import { type ReactElement } from 'react';
import type { DropzonePreviewProps } from '../types/index.js';

export function DropzonePreview({
  previewUrl,
  onClear,
  className,
}: DropzonePreviewProps): ReactElement {
  return (
    <div
      className={className}
      style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '12px',
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <img
        src={previewUrl}
        alt="Uploaded Preview"
        style={{
          maxHeight: '260px',
          maxWidth: '100%',
          objectFit: 'contain',
          borderRadius: '4px',
        }}
      />
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onClear();
        }}
        style={{
          position: 'absolute',
          top: '16px',
          right: '16px',
          backgroundColor: 'rgba(0, 0, 0, 0.65)',
          color: '#ffffff',
          border: 'none',
          borderRadius: '50%',
          width: '28px',
          height: '28px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          fontSize: '14px',
          lineHeight: 1,
        }}
        aria-label="Remove image"
      >
        ×
      </button>
    </div>
  );
}
