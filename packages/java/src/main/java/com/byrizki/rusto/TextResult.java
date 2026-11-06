package com.byrizki.rusto;

import java.util.Arrays;
import java.util.Objects;

/**
 * Recognized OCR text result containing text, confidence score, bounding polygon, frame, and optional layout metadata.
 *
 * @param text Recognized text string.
 * @param score Recognition confidence score between 0.0 and 1.0.
 * @param boxPoints Quad vertex coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
 * @param frame Axis-aligned bounding box frame.
 * @param regionId Optional layout region index when layout reading is enabled, or null.
 * @param layoutType Optional semantic layout category when layout reading is enabled, or null.
 */
public record TextResult(
    String text,
    float score,
    float[][] boxPoints,
    Frame frame,
    Integer regionId,
    String layoutType
) {
    public TextResult(String text, float score, float[][] boxPoints, Frame frame) {
        this(text, score, boxPoints, frame, null, null);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TextResult that)) return false;
        return Float.compare(score, that.score) == 0 &&
               Objects.equals(text, that.text) &&
               Arrays.deepEquals(boxPoints, that.boxPoints) &&
               Objects.equals(frame, that.frame) &&
               Objects.equals(regionId, that.regionId) &&
               Objects.equals(layoutType, that.layoutType);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(text, score, frame, regionId, layoutType);
        result = 31 * result + Arrays.deepHashCode(boxPoints);
        return result;
    }
}
