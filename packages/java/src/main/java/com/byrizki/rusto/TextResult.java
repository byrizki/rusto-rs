package com.byrizki.rusto;

import java.util.Arrays;

/**
 * Recognized OCR text result containing text, confidence score, bounding polygon, and frame.
 *
 * @param text Recognized text string.
 * @param score Recognition confidence score between 0.0 and 1.0.
 * @param boxPoints Quad vertex coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
 * @param frame Axis-aligned bounding box frame.
 */
public record TextResult(
    String text,
    float score,
    float[][] boxPoints,
    Frame frame
) {
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TextResult that)) return false;
        return Float.compare(score, that.score) == 0 &&
               text.equals(that.text) &&
               Arrays.deepEquals(boxPoints, that.boxPoints) &&
               frame.equals(that.frame);
    }

    @Override
    public int hashCode() {
        int result = text.hashCode();
        result = 31 * result + Float.hashCode(score);
        result = 31 * result + Arrays.deepHashCode(boxPoints);
        result = 31 * result + frame.hashCode();
        return result;
    }
}
