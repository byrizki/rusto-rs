import { type ReactElement } from 'react';
import type { BoundingBoxProps } from '../types/index.js';
import { scaleFrame } from '../utils/geometry.js';

export function BoundingBox({
  item,
  scaleX,
  scaleY,
  highlightColor = 'rgba(59, 130, 246, 0.4)',
  showTooltip = true,
  className,
  style,
  onClick,
}: BoundingBoxProps): ReactElement {
  const scaled = scaleFrame(item.frame, scaleX, scaleY);

  const boxStyle: React.CSSProperties = {
    position: 'absolute',
    left: `${scaled.left}px`,
    top: `${scaled.top}px`,
    width: `${scaled.width}px`,
    height: `${scaled.height}px`,
    border: `1.5px solid ${highlightColor}`,
    backgroundColor: highlightColor.replace(/[\d.]+\)$/, '0.15)'),
    cursor: onClick ? 'pointer' : 'default',
    pointerEvents: 'auto',
    boxSizing: 'border-box',
    borderRadius: '2px',
    ...style,
  };

  const tooltipStyle: React.CSSProperties = {
    position: 'absolute',
    bottom: '100%',
    left: '0',
    backgroundColor: '#1f2937',
    color: '#ffffff',
    fontSize: '11px',
    lineHeight: '1.2',
    padding: '2px 6px',
    borderRadius: '4px',
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
    zIndex: 10,
    transform: 'translateY(-2px)',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
  };

  return (
    <div
      className={className}
      style={boxStyle}
      onClick={() => onClick?.(item)}
      title={`${item.text} (${Math.round(item.score * 100)}%)`}
    >
      {showTooltip && (
        <span style={tooltipStyle}>
          {item.text} <small style={{ opacity: 0.75 }}>({Math.round(item.score * 100)}%)</small>
        </span>
      )}
    </div>
  );
}
