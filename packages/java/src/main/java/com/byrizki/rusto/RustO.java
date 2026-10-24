package com.byrizki.rusto;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * High-performance OCR engine wrapper for pure Java / JVM applications.
 */
public class RustO implements AutoCloseable {

    public static class CTextResult extends Structure {
        public static class ByReference extends CTextResult implements Structure.ByReference {}

        public String text;
        public float score;
        public float box_x1;
        public float box_y1;
        public float box_x2;
        public float box_y2;
        public float box_x3;
        public float box_y3;
        public float box_x4;
        public float box_y4;
        public float frame_width;
        public float frame_height;
        public float frame_top;
        public float frame_left;

        public CTextResult() {
            super();
        }

        public CTextResult(Pointer p) {
            super(p);
            read();
        }

        @Override
        protected List<String> getFieldOrder() {
            return List.of(
                "text", "score",
                "box_x1", "box_y1",
                "box_x2", "box_y2",
                "box_x3", "box_y3",
                "box_x4", "box_y4",
                "frame_width", "frame_height",
                "frame_top", "frame_left"
            );
        }
    }

    interface NativeLib extends Library {
        NativeLib INSTANCE = Native.load("rusto", NativeLib.class);

        Pointer rocr_initialize(String configJson);
        int rocr_detect_text_file(Pointer handle, String imagePath, String optionsJson, PointerByReference results, long[] count);
        Pointer rocr_detect_text_file_spatial(Pointer handle, String imagePath, String optionsJson);
        int rocr_detect_text_data(Pointer handle, byte[] data, long length, String optionsJson, PointerByReference results, long[] count);
        Pointer rocr_detect_text_data_spatial(Pointer handle, byte[] data, long length, String optionsJson);
        void rocr_free_string(Pointer s);
        void rocr_free_results(Pointer results, long count);
        void rocr_free(Pointer handle);
    }

    private Pointer handle;

    private RustO(Pointer handle) {
        this.handle = handle;
    }

    public static RustO initialize(String configJson) {
        if (configJson == null) {
            throw new IllegalArgumentException("configJson cannot be null");
        }
        Pointer ptr = NativeLib.INSTANCE.rocr_initialize(configJson);
        if (ptr == null) {
            throw new RuntimeException("Failed to initialize RustO OCR engine");
        }
        return new RustO(ptr);
    }

    private void ensureOpen() {
        if (handle == null) {
            throw new IllegalStateException("RustO engine instance is closed");
        }
    }

    private List<TextResult> convertResults(Pointer resultsPtr, long count) {
        if (resultsPtr == null || count <= 0) {
            return Collections.emptyList();
        }
        CTextResult template = new CTextResult();
        int structSize = template.size();
        List<TextResult> list = new ArrayList<>((int) count);

        for (int i = 0; i < count; i++) {
            Pointer itemPtr = resultsPtr.share((long) i * structSize);
            CTextResult item = new CTextResult(itemPtr);

            float[][] boxPoints = new float[][] {
                { item.box_x1, item.box_y1 },
                { item.box_x2, item.box_y2 },
                { item.box_x3, item.box_y3 },
                { item.box_x4, item.box_y4 }
            };
            Frame frame = new Frame(item.frame_width, item.frame_height, item.frame_top, item.frame_left);
            list.add(new TextResult(item.text != null ? item.text : "", item.score, boxPoints, frame));
        }
        return list;
    }

    public List<TextResult> detectTextFile(String imagePath, String optionsJson) {
        ensureOpen();
        if (imagePath == null) throw new IllegalArgumentException("imagePath cannot be null");
        String opts = optionsJson != null ? optionsJson : "{}";

        PointerByReference resultsRef = new PointerByReference();
        long[] count = new long[1];

        int status = NativeLib.INSTANCE.rocr_detect_text_file(handle, imagePath, opts, resultsRef, count);
        if (status != 0) {
            throw new RuntimeException("Text detection failed with status: " + status);
        }

        Pointer resultsPtr = resultsRef.getValue();
        try {
            return convertResults(resultsPtr, count[0]);
        } finally {
            if (resultsPtr != null) {
                NativeLib.INSTANCE.rocr_free_results(resultsPtr, count[0]);
            }
        }
    }

    public List<TextResult> detectTextData(byte[] data, String optionsJson) {
        ensureOpen();
        if (data == null || data.length == 0) {
            return Collections.emptyList();
        }
        String opts = optionsJson != null ? optionsJson : "{}";

        PointerByReference resultsRef = new PointerByReference();
        long[] count = new long[1];

        int status = NativeLib.INSTANCE.rocr_detect_text_data(handle, data, data.length, opts, resultsRef, count);
        if (status != 0) {
            throw new RuntimeException("Text detection failed with status: " + status);
        }

        Pointer resultsPtr = resultsRef.getValue();
        try {
            return convertResults(resultsPtr, count[0]);
        } finally {
            if (resultsPtr != null) {
                NativeLib.INSTANCE.rocr_free_results(resultsPtr, count[0]);
            }
        }
    }

    public String detectTextFileSpatial(String imagePath, String optionsJson) {
        ensureOpen();
        if (imagePath == null) throw new IllegalArgumentException("imagePath cannot be null");
        String opts = optionsJson != null ? optionsJson : "{}";

        Pointer strPtr = NativeLib.INSTANCE.rocr_detect_text_file_spatial(handle, imagePath, opts);
        if (strPtr == null) {
            throw new RuntimeException("Spatial text detection failed");
        }
        try {
            return strPtr.getString(0, "UTF-8");
        } finally {
            NativeLib.INSTANCE.rocr_free_string(strPtr);
        }
    }

    public String detectTextDataSpatial(byte[] data, String optionsJson) {
        ensureOpen();
        if (data == null || data.length == 0) {
            return "";
        }
        String opts = optionsJson != null ? optionsJson : "{}";

        Pointer strPtr = NativeLib.INSTANCE.rocr_detect_text_data_spatial(handle, data, data.length, opts);
        if (strPtr == null) {
            throw new RuntimeException("Spatial text detection failed");
        }
        try {
            return strPtr.getString(0, "UTF-8");
        } finally {
            NativeLib.INSTANCE.rocr_free_string(strPtr);
        }
    }

    @Override
    public synchronized void close() {
        if (handle != null) {
            NativeLib.INSTANCE.rocr_free(handle);
            handle = null;
        }
    }
}
