package com.byrizki.rusto;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class RustOTest {

    @Test
    void testInvalidConfigurationThrows() {
        assertThrows(RuntimeException.class, () -> {
            RustO.initialize("invalid-json-content");
        });
    }

    @Test
    void testNullConfigurationThrows() {
        assertThrows(IllegalArgumentException.class, () -> {
            RustO.initialize(null);
        });
    }

    @Test
    void testDataStructures() {
        Frame frame = new Frame(100f, 20f, 5f, 10f);
        assertEquals(100f, frame.width());
        assertEquals(20f, frame.height());

        float[][] box = new float[][] {
            { 10f, 5f }, { 110f, 5f }, { 110f, 25f }, { 10f, 25f }
        };
        TextResult result = new TextResult("Hello World", 0.98f, box, frame);
        assertEquals("Hello World", result.text());
        assertEquals(0.98f, result.score());
        assertEquals(frame, result.frame());
    }
}
