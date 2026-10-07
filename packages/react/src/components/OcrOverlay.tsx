import {
  useState,
  useRef,
  useCallback,
  type ReactElement,
  type SyntheticEvent,
} from 'react';
import type { OcrOverlayProps } from '../types/index.js';
import { BoundingBox } from './BoundingBox.js';
import { calculateScale } from '../utils/geometry.js';

export function OcrOverlay({
  imageSrc,
  results,
  naturalWidth: propNaturalWidth,
  naturalHeight: propNaturalHeight,
  highlightColor,
  showTooltips = true,
  className,
  style,
  onItemClick,
}: OcrOverlayProps): ReactElement {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [renderDims, setRenderDims] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  });
  const [naturalDims, setNaturalDims] = useState<{ width: number; height: number }>({
    width: propNaturalWidth || 0,
    height: propNaturalHeight || 0,
  });

  const onImageLoad = useCallback((e: SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setNaturalDims({
      width: img.naturalWidth,
      height: img.naturalHeight,
    });
    setRenderDims({
      width: img.clientWidth,
      height: img.clientHeight,
    });
  }, []);

  const effectiveNaturalWidth = propNaturalWidth || naturalDims.width;
  const effectiveNaturalHeight = propNaturalHeight || naturalDims.height;

  const { scaleX, scaleY } = calculateScale(
    effectiveNaturalWidth,
    effectiveNaturalHeight,
    renderDims.width || effectiveNaturalWidth,
    renderDims.height || effectiveNaturalHeight
  );

  return (
    <div
      className={className}
      style={{
        position: 'relative',
        display: 'inline-block',
        overflow: 'hidden',
        ...style,
      }}
    >
      <img
        ref={imgRef}
        src={imageSrc}
        alt="OCR Target"
        onLoad={onImageLoad}
        style={{
          display: 'block',
          maxWidth: '100%',
          height: 'auto',
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          pointerEvents: 'none',
        }}
      >
        {results.map((item, idx) => (
          <BoundingBox
            key={`${item.text}-${idx}`}
            item={item}
            scaleX={scaleX}
            scaleY={scaleY}
            highlightColor={highlightColor}
            showTooltip={showTooltips}
            onClick={onItemClick}
          />
        ))}
      </div>
    </div>
  );
}
