import { type ReactElement } from 'react';
import type { DropzonePromptProps } from '../types/index.js';

export function DropzonePrompt({
  title = 'Click or drag image here to scan',
  subtitle = 'Supports PNG, JPEG, WebP, and document scans',
  icon,
  className,
}: DropzonePromptProps): ReactElement {
  return (
    <div
      className={className}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '32px 16px',
        textAlign: 'center',
        pointerEvents: 'none',
      }}
    >
      {icon || (
        <svg
          style={{ width: '40px', height: '40px', color: '#6b7280', marginBottom: '12px' }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
      )}
      <p style={{ margin: '0 0 4px', fontSize: '14px', fontWeight: 600, color: '#374151' }}>
        {title}
      </p>
      <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>
        {subtitle}
      </p>
    </div>
  );
}
