package com.byrizki.rusto;

import java.util.Arrays;

/**
 * Recognized text result with confidence score and bounding coordinates.
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
