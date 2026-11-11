package com.byrizki.rusto;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Pattern;

/**
 * High-performance OCR engine wrapper for pure Java / JVM applications.
 */
public class RustO implements AutoCloseable {

    private static final Pattern SPATIAL_OUTPUT_PATTERN =
        Pattern.compile("\"output\"\\s*:\\s*\"spatial\"");
    private static final Pattern TSV_OUTPUT_PATTERN =
        Pattern.compile("\"output\"\\s*:\\s*\"tsv\"");

    /**
     * Unified OCR result holding either structured recognition items or a 2D spatial layout string.
     */
    public static final class DetectTextResult {
        private final List<TextResult> results;
        private final String text;
        private final boolean spatial;

        private DetectTextResult(List<TextResult> results, String text, boolean spatial) {
            this.results = results != null ? Collections.unmodifiableList(results) : Collections.emptyList();
            this.text = text != null ? text : "";
            this.spatial = spatial;
        }

        public static DetectTextResult structured(List<TextResult> results) {
            return new DetectTextResult(results, "", false);
        }

        public static DetectTextResult spatial(String text) {
            return new DetectTextResult(Collections.emptyList(), text, true);
        }

        public boolean isSpatial() {
            return spatial;
        }

        public boolean isStructured() {
            return !spatial;
        }

        public List<TextResult> results() {
            return results;
        }

        public List<TextResult> getResults() {
            return results;
        }

        public String text() {
            return text;
        }

        public String getText() {
            return text;
        }
    }

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
        public int region_id;
        public int layout_type;

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
                "frame_top", "frame_left",
                "region_id", "layout_type"
            );
        }
    }

    interface NativeLib extends Library {
        NativeLib INSTANCE = Native.load("rusto", NativeLib.class);

        Pointer rocr_initialize(String configJson);
        int rocr_detect_text_file(Pointer handle, String imagePath, String optionsJson, PointerByReference results, long[] count);
        Pointer rocr_detect_text_file_spatial(Pointer handle, String imagePath, String optionsJson);
        Pointer rocr_detect_text_file_tsv(Pointer handle, String imagePath, String optionsJson);
        int rocr_detect_text_data(Pointer handle, byte[] data, long length, String optionsJson, PointerByReference results, long[] count);
        Pointer rocr_detect_text_data_spatial(Pointer handle, byte[] data, long length, String optionsJson);
        Pointer rocr_detect_text_data_tsv(Pointer handle, byte[] data, long length, String optionsJson);
        int rocr_detect_text_rgb(Pointer handle, byte[] pixels, long imageLen, int width, int height, long stride, String optionsJson, PointerByReference results, long[] count);
        Pointer rocr_detect_text_rgb_spatial(Pointer handle, byte[] pixels, long imageLen, int width, int height, long stride, String optionsJson);
        Pointer rocr_detect_text_rgb_tsv(Pointer handle, byte[] pixels, long imageLen, int width, int height, long stride, String optionsJson);
        void rocr_free_string(Pointer s);
        void rocr_free_results(Pointer results, long count);
        void rocr_free(Pointer handle);
        String rocr_version();
        int rocr_is_ready(Pointer handle);
        Pointer rocr_download_model(String model, String lang, String cacheDir);
    }

    private Pointer handle;

    private RustO(Pointer handle) {
        this.handle = handle;
    }

    /**
     * Initializes the RustO OCR engine with default PP-OCRv6 Tiny (FP32) configuration (auto-downloads models if needed).
     *
     * @return Initialized RustO engine instance.
     */
    public static RustO initialize() {
        return initialize("{}");
    }

    /**
     * Initializes the RustO OCR engine instance with a JSON configuration string.
     *
     * @param configJson JSON string containing model paths and runtime configuration.
     * @return Initialized RustO engine instance.
     * @throws IllegalArgumentException if configJson is null.
     * @throws RuntimeException if native engine initialization fails.
     */
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

    /**
     * Returns the engine library version string.
     */
    public static String version() {
        return NativeLib.INSTANCE.rocr_version();
    }

    /**
     * Returns the engine library version string.
     */
    public String getVersion() {
        return version();
    }

    /**
     * Checks if the engine instance is initialized and ready for inference.
     */
    public boolean isReady() {
        return handle != null && NativeLib.INSTANCE.rocr_is_ready(handle) != 0;
    }

    /**
     * Downloads model assets for a specific OCR pipeline ahead of time.
     *
     * @param model Model identifier (e.g. "ppocrv6-tiny", "indraocr"), or null for default.
     * @param lang Language code (e.g. "latin"), or null for default.
     * @param cacheDir Destination cache directory, or null for default.
     * @return List of downloaded model asset file paths.
     */
    public static List<String> downloadModel(String model, String lang, String cacheDir) {
        Pointer ptr = NativeLib.INSTANCE.rocr_download_model(model, lang, cacheDir);
        if (ptr == null) {
            throw new RuntimeException("Failed to download model assets");
        }
        try {
            String json = ptr.getString(0, "UTF-8");
            List<String> list = new ArrayList<>();
            java.util.regex.Matcher m = java.util.regex.Pattern.compile("\"([^\"]*)\"").matcher(json);
            while (m.find()) {
                list.add(m.group(1));
            }
            return Collections.unmodifiableList(list);
        } finally {
            NativeLib.INSTANCE.rocr_free_string(ptr);
        }
    }

    /**
     * Downloads default PP-OCRv6 Tiny (FP32) model assets ahead of time.
     */
    public static List<String> downloadModel() {
        return downloadModel("ppocrv6-tiny", null, null);
    }

    private void ensureOpen() {
        if (handle == null) {
            throw new IllegalStateException("RustO engine instance is closed");
        }
    }

    private static String formatLayoutType(int code) {
        return switch (code) {
            case 0 -> "text";
            case 1 -> "title";
            case 2 -> "figure";
            case 3 -> "figureCaption";
            case 4 -> "table";
            case 5 -> "tableCaption";
            case 6 -> "header";
            case 7 -> "footer";
            case 8 -> "reference";
            case 9 -> "equation";
            default -> null;
        };
    }

    private static boolean isSpatialOptionsJson(String optionsJson) {
        return optionsJson != null && SPATIAL_OUTPUT_PATTERN.matcher(optionsJson).find();
    }

    private static boolean isTsvOptionsJson(String optionsJson) {
        return optionsJson != null && TSV_OUTPUT_PATTERN.matcher(optionsJson).find();
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
            Integer regionId = item.region_id >= 0 ? item.region_id : null;
            String layoutType = formatLayoutType(item.layout_type);
            list.add(new TextResult(
                item.text != null ? item.text : "",
                item.score,
                boxPoints,
                frame,
                regionId,
                layoutType
            ));
        }
        return list;
    }

    private List<TextResult> detectTextFile(String imagePath, String optionsJson) {
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

    private List<TextResult> detectTextData(byte[] data, String optionsJson) {
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

    private String detectTextFileSpatial(String imagePath, String optionsJson) {
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

    private String detectTextFileTsv(String imagePath, String optionsJson) {
        ensureOpen();
        if (imagePath == null) throw new IllegalArgumentException("imagePath cannot be null");
        String opts = optionsJson != null ? optionsJson : "{}";

        Pointer strPtr = NativeLib.INSTANCE.rocr_detect_text_file_tsv(handle, imagePath, opts);
        if (strPtr == null) {
            throw new RuntimeException("TSV text detection failed");
        }
        try {
            return strPtr.getString(0, "UTF-8");
        } finally {
            NativeLib.INSTANCE.rocr_free_string(strPtr);
        }
    }

    private String detectTextDataSpatial(byte[] data, String optionsJson) {
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

    private String detectTextDataTsv(byte[] data, String optionsJson) {
        ensureOpen();
        if (data == null || data.length == 0) {
            return "";
        }
        String opts = optionsJson != null ? optionsJson : "{}";

        Pointer strPtr = NativeLib.INSTANCE.rocr_detect_text_data_tsv(handle, data, data.length, opts);
        if (strPtr == null) {
            throw new RuntimeException("TSV text detection failed");
        }
        try {
            return strPtr.getString(0, "UTF-8");
        } finally {
            NativeLib.INSTANCE.rocr_free_string(strPtr);
        }
    }

    private List<TextResult> detectTextRGBStructured(byte[] pixels, int width, int height, long stride, String optionsJson) {
        ensureOpen();
        if (pixels == null || pixels.length == 0) {
            return Collections.emptyList();
        }
        String opts = optionsJson != null ? optionsJson : "{}";
        PointerByReference resultsRef = new PointerByReference();
        long[] count = new long[1];

        int status = NativeLib.INSTANCE.rocr_detect_text_rgb(
            handle, pixels, pixels.length, width, height, stride, opts, resultsRef, count
        );
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

    private String detectTextRGBSpatial(byte[] pixels, int width, int height, long stride, String optionsJson) {
        ensureOpen();
        if (pixels == null || pixels.length == 0) {
            return "";
        }
        String opts = optionsJson != null ? optionsJson : "{}";

        Pointer strPtr = NativeLib.INSTANCE.rocr_detect_text_rgb_spatial(
            handle, pixels, pixels.length, width, height, stride, opts
        );
        if (strPtr == null) {
            throw new RuntimeException("Spatial text detection failed");
        }
        try {
            return strPtr.getString(0, "UTF-8");
        } finally {
            NativeLib.INSTANCE.rocr_free_string(strPtr);
        }
    }

    private String detectTextRGBTsv(byte[] pixels, int width, int height, long stride, String optionsJson) {
        ensureOpen();
        if (pixels == null || pixels.length == 0) {
            return "";
        }
        String opts = optionsJson != null ? optionsJson : "{}";

        Pointer strPtr = NativeLib.INSTANCE.rocr_detect_text_rgb_tsv(
            handle, pixels, pixels.length, width, height, stride, opts
        );
        if (strPtr == null) {
            throw new RuntimeException("TSV text detection failed");
        }
        try {
            return strPtr.getString(0, "UTF-8");
        } finally {
            NativeLib.INSTANCE.rocr_free_string(strPtr);
        }
    }

    /**
     * Detects text in an image file at the specified filesystem path with explicit JSON options.
     */
    public DetectTextResult detectText(String imagePath, String optionsJson) {
        if (isTsvOptionsJson(optionsJson)) {
            return DetectTextResult.spatial(detectTextFileTsv(imagePath, optionsJson));
        }
        if (isSpatialOptionsJson(optionsJson)) {
            return DetectTextResult.spatial(detectTextFileSpatial(imagePath, optionsJson));
        }
        return DetectTextResult.structured(detectTextFile(imagePath, optionsJson));
    }

    /**
     * Starts a fluent builder for chaining detection options with image path.
     */
    public DetectTextBuilder detectText(String imagePath) {
        return new DetectTextBuilder(this).image(imagePath);
    }

    /**
     * Detects text from an in-memory encoded image byte array with explicit JSON options.
     */
    public DetectTextResult detectText(byte[] data, String optionsJson) {
        if (isTsvOptionsJson(optionsJson)) {
            return DetectTextResult.spatial(detectTextDataTsv(data, optionsJson));
        }
        if (isSpatialOptionsJson(optionsJson)) {
            return DetectTextResult.spatial(detectTextDataSpatial(data, optionsJson));
        }
        return DetectTextResult.structured(detectTextData(data, optionsJson));
    }

    /**
     * Starts a fluent builder for chaining detection options with image data.
     */
    public DetectTextBuilder detectText(byte[] data) {
        return new DetectTextBuilder(this).data(data);
    }

    /**
     * Detects text from raw decoded interleaved RGB8 pixel bytes with explicit JSON options.
     */
    public DetectTextResult detectTextRgb(byte[] pixels, int width, int height, long stride, String optionsJson) {
        if (isTsvOptionsJson(optionsJson)) {
            return DetectTextResult.spatial(detectTextRGBTsv(pixels, width, height, stride, optionsJson));
        }
        if (isSpatialOptionsJson(optionsJson)) {
            return DetectTextResult.spatial(detectTextRGBSpatial(pixels, width, height, stride, optionsJson));
        }
        return DetectTextResult.structured(detectTextRGBStructured(pixels, width, height, stride, optionsJson));
    }

    /**
     * Starts a fluent builder for chaining detection options.
     */
    public DetectTextBuilder detectText() {
        return new DetectTextBuilder(this);
    }

    /**
     * Starts a fluent builder for chaining detection options.
     */
    public DetectTextBuilder detectTextBuilder() {
        return new DetectTextBuilder(this);
    }

    /**
     * Fluent builder for chaining text detection options and running inference.
     */
    public static class DetectTextBuilder {
        private final RustO engine;
        private String imagePath;
        private byte[] data;
        private boolean hasData;
        private byte[] rgbPixels;
        private boolean hasRgb;
        private int rgbWidth;
        private int rgbHeight;
        private long rgbStride;

        private String output = "lines";
        private Boolean autoUnwarp;
        private Boolean layoutReading;
        private Boolean orientationDetection;
        private Boolean lineClassification;

        private Double descreenStrength;
        private Integer targetMaxSide;
        private Double cropPaddingX;
        private Double cropPaddingY;

        private Integer limitSideLen;
        private String limitType;

        private Double threshold;
        private Double boxThreshold;
        private Double unclipRatio;
        private Boolean useDilation;
        private String scoreMode;
        private Integer maxCandidates;
        private Double minHeight;
        private Double minSideLen;
        private Double maxSideLen;
        private Double widthHeightRatio;

        private Double lineYThreshold;
        private Double wordXThreshold;

        private Integer batchSize;

        private Double scoreThreshold;

        private String customJson;

        public DetectTextBuilder(RustO engine) {
            this.engine = engine;
        }

        public DetectTextBuilder image(String imagePath) {
            this.imagePath = imagePath;
            this.hasData = false;
            this.hasRgb = false;
            return this;
        }

        public DetectTextBuilder data(byte[] data) {
            this.data = data;
            this.hasData = true;
            this.imagePath = null;
            this.hasRgb = false;
            return this;
        }

        public DetectTextBuilder rgb(byte[] pixels, int width, int height) {
            return rgb(pixels, width, height, (long) width * 3L);
        }

        public DetectTextBuilder rgb(byte[] pixels, int width, int height, long stride) {
            this.rgbPixels = pixels;
            this.hasRgb = true;
            this.rgbWidth = width;
            this.rgbHeight = height;
            this.rgbStride = stride;
            this.imagePath = null;
            this.hasData = false;
            return this;
        }

        public DetectTextBuilder withAutoUnwarp(boolean enabled) {
            this.autoUnwarp = enabled;
            return this;
        }

        public DetectTextBuilder withAutoUnwarp() {
            return withAutoUnwarp(true);
        }

        public DetectTextBuilder withLayoutReading(boolean enabled) {
            this.layoutReading = enabled;
            return this;
        }

        public DetectTextBuilder withLayoutReading() {
            return withLayoutReading(true);
        }

        public DetectTextBuilder withOrientationDetection(boolean enabled) {
            this.orientationDetection = enabled;
            return this;
        }

        public DetectTextBuilder withOrientationDetection() {
            return withOrientationDetection(true);
        }

        public DetectTextBuilder withLineClassification(boolean enabled) {
            this.lineClassification = enabled;
            return this;
        }

        public DetectTextBuilder withLineClassification() {
            return withLineClassification(true);
        }

        public DetectTextBuilder withPreDetection(Double descreenStrength, Integer targetMaxSide, Double cropPaddingX, Double cropPaddingY) {
            this.descreenStrength = descreenStrength;
            this.targetMaxSide = targetMaxSide;
            this.cropPaddingX = cropPaddingX;
            this.cropPaddingY = cropPaddingY;
            return this;
        }

        public DetectTextBuilder withDetection(Integer limitSideLen, String limitType) {
            this.limitSideLen = limitSideLen;
            this.limitType = limitType;
            return this;
        }

        public DetectTextBuilder withPostDetection(Double threshold, Double boxThreshold, Double unclipRatio, Boolean useDilation) {
            this.threshold = threshold;
            this.boxThreshold = boxThreshold;
            this.unclipRatio = unclipRatio;
            this.useDilation = useDilation;
            return this;
        }

        public DetectTextBuilder withPreRecognition(Double lineYThreshold, Double wordXThreshold) {
            this.lineYThreshold = lineYThreshold;
            this.wordXThreshold = wordXThreshold;
            return this;
        }

        public DetectTextBuilder withRecognition(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public DetectTextBuilder withPostRecognition(double scoreThreshold) {
            this.scoreThreshold = scoreThreshold;
            return this;
        }

        public DetectTextBuilder asLines() {
            this.output = "lines";
            return this;
        }

        public DetectTextBuilder asWords() {
            this.output = "words";
            return this;
        }

        public DetectTextBuilder asSpatial() {
            this.output = "spatial";
            return this;
        }

        public DetectTextBuilder asTsv() {
            this.output = "tsv";
            return this;
        }

        public DetectTextBuilder withOutput(String output) {
            this.output = output;
            return this;
        }

        public DetectTextBuilder withOptions(String optionsJson) {
            this.customJson = optionsJson;
            return this;
        }

        public String buildOptionsJson() {
            if (customJson != null && !customJson.trim().isEmpty()) {
                return customJson;
            }
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            if (output != null) {
                sb.append("\"output\":\"").append(output).append("\"");
                first = false;
            }
            if (autoUnwarp != null) {
                if (!first) sb.append(",");
                sb.append("\"autoUnwarp\":").append(autoUnwarp);
                first = false;
            }
            if (layoutReading != null) {
                if (!first) sb.append(",");
                sb.append("\"layoutReading\":").append(layoutReading);
                first = false;
            }
            if (orientationDetection != null) {
                if (!first) sb.append(",");
                sb.append("\"orientationDetection\":").append(orientationDetection);
                first = false;
            }
            if (lineClassification != null) {
                if (!first) sb.append(",");
                sb.append("\"lineClassification\":").append(lineClassification);
                first = false;
            }
            if (descreenStrength != null || targetMaxSide != null || cropPaddingX != null || cropPaddingY != null) {
                if (!first) sb.append(",");
                sb.append("\"preDetection\":{");
                boolean pf = true;
                if (descreenStrength != null) { sb.append("\"descreenStrength\":").append(descreenStrength); pf = false; }
                if (targetMaxSide != null) { if (!pf) sb.append(","); sb.append("\"targetMaxSide\":").append(targetMaxSide); pf = false; }
                if (cropPaddingX != null) { if (!pf) sb.append(","); sb.append("\"cropPaddingX\":").append(cropPaddingX); pf = false; }
                if (cropPaddingY != null) { if (!pf) sb.append(","); sb.append("\"cropPaddingY\":").append(cropPaddingY); }
                sb.append("}");
                first = false;
            }
            if (limitSideLen != null || limitType != null) {
                if (!first) sb.append(",");
                sb.append("\"detection\":{");
                boolean df = true;
                if (limitSideLen != null) { sb.append("\"limitSideLen\":").append(limitSideLen); df = false; }
                if (limitType != null) { if (!df) sb.append(","); sb.append("\"limitType\":\"").append(limitType).append("\""); }
                sb.append("}");
                first = false;
            }
            if (threshold != null || boxThreshold != null || unclipRatio != null || useDilation != null || scoreMode != null || maxCandidates != null || minHeight != null || minSideLen != null || maxSideLen != null || widthHeightRatio != null) {
                if (!first) sb.append(",");
                sb.append("\"postDetection\":{");
                boolean pdf = true;
                if (threshold != null) { sb.append("\"threshold\":").append(threshold); pdf = false; }
                if (boxThreshold != null) { if (!pdf) sb.append(","); sb.append("\"boxThreshold\":").append(boxThreshold); pdf = false; }
                if (unclipRatio != null) { if (!pdf) sb.append(","); sb.append("\"unclipRatio\":").append(unclipRatio); pdf = false; }
                if (useDilation != null) { if (!pdf) sb.append(","); sb.append("\"useDilation\":").append(useDilation); pdf = false; }
                if (scoreMode != null) { if (!pdf) sb.append(","); sb.append("\"scoreMode\":\"").append(scoreMode).append("\""); pdf = false; }
                if (maxCandidates != null) { if (!pdf) sb.append(","); sb.append("\"maxCandidates\":").append(maxCandidates); pdf = false; }
                if (minHeight != null) { if (!pdf) sb.append(","); sb.append("\"minHeight\":").append(minHeight); pdf = false; }
                if (minSideLen != null) { if (!pdf) sb.append(","); sb.append("\"minSideLen\":").append(minSideLen); pdf = false; }
                if (maxSideLen != null) { if (!pdf) sb.append(","); sb.append("\"maxSideLen\":").append(maxSideLen); pdf = false; }
                if (widthHeightRatio != null) { if (!pdf) sb.append(","); sb.append("\"widthHeightRatio\":").append(widthHeightRatio); }
                sb.append("}");
                first = false;
            }
            if (lineYThreshold != null || wordXThreshold != null) {
                if (!first) sb.append(",");
                sb.append("\"preRecognition\":{");
                boolean prf = true;
                if (lineYThreshold != null) { sb.append("\"lineYThreshold\":").append(lineYThreshold); prf = false; }
                if (wordXThreshold != null) { if (!prf) sb.append(","); sb.append("\"wordXThreshold\":").append(wordXThreshold); }
                sb.append("}");
                first = false;
            }
            if (batchSize != null) {
                if (!first) sb.append(",");
                sb.append("\"recognition\":{\"batchSize\":").append(batchSize).append("}");
                first = false;
            }
            if (scoreThreshold != null) {
                if (!first) sb.append(",");
                sb.append("\"postRecognition\":{\"scoreThreshold\":").append(scoreThreshold).append("}");
            }
            sb.append("}");
            return sb.toString();
        }

        public DetectTextResult run() {
            String json = buildOptionsJson();
            boolean tsvMode = customJson != null && !customJson.trim().isEmpty()
                ? isTsvOptionsJson(json)
                : "tsv".equals(output);

            if (tsvMode) {
                if (imagePath != null) {
                    return DetectTextResult.spatial(engine.detectTextFileTsv(imagePath, json));
                }
                if (hasData) {
                    return DetectTextResult.spatial(engine.detectTextDataTsv(data, json));
                }
                if (hasRgb) {
                    return DetectTextResult.spatial(engine.detectTextRGBTsv(rgbPixels, rgbWidth, rgbHeight, rgbStride, json));
                }
                throw new IllegalStateException("No image source provided to DetectTextBuilder");
            }

            boolean spatialMode = customJson != null && !customJson.trim().isEmpty()
                ? isSpatialOptionsJson(json)
                : "spatial".equals(output);

            if (spatialMode) {
                if (imagePath != null) {
                    return DetectTextResult.spatial(engine.detectTextFileSpatial(imagePath, json));
                }
                if (hasData) {
                    return DetectTextResult.spatial(engine.detectTextDataSpatial(data, json));
                }
                if (hasRgb) {
                    return DetectTextResult.spatial(engine.detectTextRGBSpatial(rgbPixels, rgbWidth, rgbHeight, rgbStride, json));
                }
                throw new IllegalStateException("No image source provided to DetectTextBuilder");
            }

            if (imagePath != null) {
                return DetectTextResult.structured(engine.detectTextFile(imagePath, json));
            }
            if (hasData) {
                return DetectTextResult.structured(engine.detectTextData(data, json));
            }
            if (hasRgb) {
                return DetectTextResult.structured(engine.detectTextRGBStructured(rgbPixels, rgbWidth, rgbHeight, rgbStride, json));
            }
            throw new IllegalStateException("No image source provided to DetectTextBuilder");
        }
    }

    /**
     * Releases the native RustO OCR engine handle and associated model memory.
     */
    @Override
    public synchronized void close() {
        if (handle != null) {
            NativeLib.INSTANCE.rocr_free(handle);
            handle = null;
        }
    }
}
