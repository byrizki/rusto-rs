package com.byrizki.rusto;

import org.junit.jupiter.api.Test;
import java.io.File;
import java.util.List;
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

    @Test
    void testRealOcrOnExample1Image() {
        File det = new File("../../models/PPOCR_v6/det.rten");
        File rec = new File("../../models/PPOCR_v6/rec.rten");
        File dict = new File("../../models/PPOCR_v6/dict.txt");
        File img = new File("../../models/test_images/example1.png");

        if (!det.exists() || !img.exists()) {
            return;
        }

        String config = String.format(
            "{\"detection\":{\"model_path\":\"%s\"},\"recognition\":{\"model_path\":\"%s\",\"dict_path\":\"%s\"}}",
            det.getAbsolutePath(), rec.getAbsolutePath(), dict.getAbsolutePath()
        );

        try (RustO engine = RustO.initialize(config)) {
            List<TextResult> results = engine.detectTextFile(img.getAbsolutePath(), "{}");
            assertNotNull(results);
            assertFalse(results.isEmpty());
            assertTrue(results.stream().anyMatch(r -> r.text().contains("INVOICE")));

            String spatial = engine.detectTextFileSpatial(img.getAbsolutePath(), "{}");
            assertNotNull(spatial);
            assertTrue(spatial.contains("INVOICE"));
        }
    }
}
