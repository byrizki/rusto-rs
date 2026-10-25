package com.byrizki.rusto;

/**
 * Axis-aligned bounding box frame with position and size in pixels.
 *
 * @param width Width of the bounding box.
 * @param height Height of the bounding box.
 * @param top Top vertical coordinate (Y).
 * @param left Left horizontal coordinate (X).
 */
public record Frame(float width, float height, float top, float left) {}
