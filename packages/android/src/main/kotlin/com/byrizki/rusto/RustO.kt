package com.byrizki.rusto

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.net.URLDecoder

/**
 * 2D coordinate point representing vertex location in pixels.
 */
data class Point2D(val x: Float, val y: Float)

/**
 * Axis-aligned bounding box frame with position and size in pixels.
 */
data class Frame(
    val width: Float,
    val height: Float,
    val top: Float,
    val left: Float
)

/**
 * Recognized OCR text result containing text, confidence score, bounding polygon, and frame.
 */
data class TextResult(
    val text: String,
    val score: Float,
    val boxPoints: List<Point2D>,
    val frame: Frame = run {
        val minX = boxPoints.minOfOrNull { it.x } ?: 0f
        val maxX = boxPoints.maxOfOrNull { it.x } ?: 0f
        val minY = boxPoints.minOfOrNull { it.y } ?: 0f
        val maxY = boxPoints.maxOfOrNull { it.y } ?: 0f
        Frame(width = maxX - minX, height = maxY - minY, top = minY, left = minX)
    },
    val regionId: Int? = null,
    val layoutType: String? = null
)

/**
 * Text detection model configuration.
 */
data class DetectionConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null,
    val thresh: Float? = null,
    val boxThresh: Float? = null,
    val unclipRatio: Float? = null,
    val limitSideLen: Int? = null,
    val limitType: String? = null,
    val useDilation: Boolean? = null,
    val scoreMode: String? = null
)

/**
 * Text recognition model configuration.
 *
 * @property enabled Whether recognition stage is enabled.
 * @property modelPath Filesystem path to recognition model (.rten or .onnx).
 * @property dictPath Filesystem path to dictionary vocabulary file (.txt).
 * @property scoreThresh Minimum confidence score threshold.
 * @property returnWordBox Whether to return word-level bounding boxes.
 * @property returnSingleCharBox Whether to return single-character bounding boxes.
 * @property recBatchNum Recognition batch size. Default: 6.
 * @property recWorkers Number of parallel worker threads for recognition (`null` = auto).
 * @property maxRecWorkers Maximum worker thread count clamp for recognition (`null` = engine default 8).
 */
data class RecognitionConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null,
    val dictPath: String? = null,
    val scoreThresh: Float? = null,
    val returnWordBox: Boolean? = null,
    val returnSingleCharBox: Boolean? = null,
    val recBatchNum: Int? = null,
    val recWorkers: Int? = null,
    val maxRecWorkers: Int? = null
)

/**
 * Line Classification Configuration (CLS)
 * NOTE: Text line orientation classifier (180° rotation) is ONLY available on PP-OCRv4 and PP-OCRv5.
 */
data class ClassificationConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null,
    val thresh: Float? = null
)

/**
 * Document orientation angle classifier configuration (0°, 90°, 180°, 270°).
 */
data class OrientationConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null,
    val thresh: Float? = null
)

/**
 * Document unwarping / dewarping model configuration.
 */
data class UnwarpConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null
)

/**
 * Document layout analysis and spatial reconstruction configuration.
 */
data class LayoutConfig(
    val yThresholdMultiplier: Float? = null,
    val xThresholdMultiplier: Float? = null,
    val enabled: Boolean? = null,
    val modelPath: String? = null,
    val confThresh: Float? = null,
    val targetSize: Int? = null,
)

/**
 * Configuration options for initializing the RustO engine on Android.
 */
data class InitializeConfig(
    val model: String? = null,
    val modelPath: String? = null,
    val lang: String? = null,
    val cacheDir: String? = null,
    val enableReadingOrder: Boolean? = null,
    val layoutVariant: String? = null,
    val template: String? = null,
    val detection: DetectionConfig? = null,
    val recognition: RecognitionConfig? = null,
    val classification: ClassificationConfig? = null,
    val orientation: OrientationConfig? = null,
    val unwarp: UnwarpConfig? = null,
    val layout: LayoutConfig? = null
) {
    fun toJson(context: Context): String {
        val json = JSONObject()
        model?.let { json.put("model", it) }
        modelPath?.let { json.put("modelPath", it) }
        lang?.let { json.put("lang", it) }
        val resolvedCacheDir = cacheDir ?: File(context.cacheDir, "rusto/models").absolutePath
        json.put("cacheDir", resolvedCacheDir)
        enableReadingOrder?.let { json.put("enableReadingOrder", it) }
        layoutVariant?.let { json.put("layoutVariant", it) }
        template?.let { json.put("template", it) }

        val hasModelDir = model != null || modelPath != null
        val hasBundledDefaults = !hasModelDir && detection == null && recognition == null &&
            (hasAsset(context, "det.mnn") || hasAsset(context, "det.rten") || hasAsset(context, "det.onnx"))

        if (detection != null || hasBundledDefaults) {
            val detObj = JSONObject()
            val detPath = detection?.modelPath ?: if (hasAsset(context, "det.mnn")) "det.mnn" else "det.rten"
            detObj.put("modelPath", resolveModel(context, detPath))
            detection?.enabled?.let { detObj.put("enabled", it) }
            detection?.thresh?.let { detObj.put("thresh", it.toDouble()) }
            detection?.boxThresh?.let { detObj.put("boxThresh", it.toDouble()) }
            detection?.unclipRatio?.let { detObj.put("unclipRatio", it.toDouble()) }
            detection?.limitSideLen?.let { detObj.put("limitSideLen", it) }
            detection?.limitType?.let { detObj.put("limitType", it) }
            detection?.useDilation?.let { detObj.put("useDilation", it) }
            detection?.scoreMode?.let { detObj.put("scoreMode", it) }
            json.put("detection", detObj)
        }

        if (recognition != null || hasBundledDefaults) {
            val recObj = JSONObject()
            val recPath = recognition?.modelPath ?: if (hasAsset(context, "rec.mnn")) "rec.mnn" else "rec.rten"
            val dictPath = recognition?.dictPath ?: if (hasAsset(context, "meta.json")) "meta.json" else "dict.txt"
            recObj.put("modelPath", resolveModel(context, recPath))
            recObj.put("dictPath", resolveModel(context, dictPath))
            recognition?.enabled?.let { recObj.put("enabled", it) }
            recognition?.scoreThresh?.let { recObj.put("scoreThresh", it.toDouble()) }
            recognition?.returnWordBox?.let { recObj.put("returnWordBox", it) }
            recognition?.returnSingleCharBox?.let { recObj.put("returnSingleCharBox", it) }
            recognition?.recBatchNum?.let { recObj.put("recBatchNum", it) }
            recognition?.recWorkers?.let { recObj.put("recWorkers", it) }
            recognition?.maxRecWorkers?.let { recObj.put("maxRecWorkers", it) }
            json.put("recognition", recObj)
        }

        classification?.let { cls ->
            val clsObj = JSONObject()
            cls.enabled?.let { clsObj.put("enabled", it) }
            cls.modelPath?.let { clsObj.put("modelPath", resolveModel(context, it)) }
            cls.thresh?.let { clsObj.put("thresh", it.toDouble()) }
            if (clsObj.length() > 0) json.put("classification", clsObj)
        }

        orientation?.let { orient ->
            val orientObj = JSONObject()
            orient.enabled?.let { orientObj.put("enabled", it) }
            orient.modelPath?.let { orientObj.put("modelPath", resolveModel(context, it)) }
            orient.thresh?.let { orientObj.put("thresh", it.toDouble()) }
            if (orientObj.length() > 0) json.put("orientation", orientObj)
        }

        unwarp?.let { unw ->
            val unwObj = JSONObject()
            unw.enabled?.let { unwObj.put("enabled", it) }
            unw.modelPath?.let { unwObj.put("modelPath", resolveModel(context, it)) }
            if (unwObj.length() > 0) json.put("unwarp", unwObj)
        }

        layout?.let { lay ->
            val layObj = JSONObject()
            lay.enabled?.let { layObj.put("enabled", it) }
            lay.modelPath?.let { layObj.put("modelPath", resolveModel(context, it)) }
            lay.confThresh?.let { layObj.put("confThresh", it.toDouble()) }
            lay.targetSize?.let { layObj.put("targetSize", it) }
            lay.yThresholdMultiplier?.let { layObj.put("yThresholdMultiplier", it.toDouble()) }
            lay.xThresholdMultiplier?.let { layObj.put("xThresholdMultiplier", it.toDouble()) }
            if (layObj.length() > 0) json.put("layout", layObj)
        }

        return json.toString()
    }

    private fun hasAsset(context: Context, assetName: String): Boolean {
        return try {
            context.assets.open(assetName).close()
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun resolveModel(context: Context, pathOrAsset: String): String {
        val file = File(pathOrAsset)
        if (file.exists()) return file.absolutePath
        return RustO.copyAssetToCache(context, pathOrAsset)
    }

    companion object {
        fun forModel(model: String = "ppocrv6-tiny", lang: String? = null, modelPath: String? = null): InitializeConfig =
            InitializeConfig(model = model, lang = lang, modelPath = modelPath)

        fun indra(detModelPath: String? = null, recModelPath: String? = null, dictPath: String? = null): InitializeConfig =
            preset("indra", detModelPath, recModelPath, dictPath, recBatchNum = 1)

        fun ppv6(detModelPath: String? = null, recModelPath: String? = null, dictPath: String? = null): InitializeConfig =
            preset("ppv6", detModelPath, recModelPath, dictPath)

        fun ppv5(detModelPath: String? = null, recModelPath: String? = null, dictPath: String? = null): InitializeConfig =
            preset("ppv5", detModelPath, recModelPath, dictPath)

        fun ppv4(detModelPath: String? = null, recModelPath: String? = null, dictPath: String? = null): InitializeConfig =
            preset("ppv4", detModelPath, recModelPath, dictPath)

        fun ppv3(detModelPath: String? = null, recModelPath: String? = null, dictPath: String? = null): InitializeConfig =
            preset("ppv3", detModelPath, recModelPath, dictPath)

        private fun preset(
            template: String,
            detModelPath: String?,
            recModelPath: String?,
            dictPath: String?,
            recBatchNum: Int? = null,
        ): InitializeConfig {
            if (detModelPath == null && recModelPath == null && dictPath == null) {
                return InitializeConfig(template = template)
            }
            return InitializeConfig(
                template = template,
                detection = detModelPath?.let { DetectionConfig(enabled = true, modelPath = it) },
                recognition = if (recModelPath != null || dictPath != null) {
                    RecognitionConfig(enabled = true, modelPath = recModelPath, dictPath = dictPath, recBatchNum = recBatchNum)
                } else null,
            )
        }
    }
}

class RustOException(message: String) : Exception(message)

/**
 * Output format granularity:
 * - LINES: Group recognized tokens into lines.
 * - WORDS: Return individual word-level tokens.
 * - SPATIAL: Return a formatted 2D text layout preserving spatial arrangement.
 */
enum class OutputGranularity(val wireValue: String) {
    LINES("lines"), WORDS("words"), SPATIAL("spatial"), TSV("tsv");
}

/**
 * Phase 1: Pre-detection options (descreen, downscale, padding).
 */
data class PreDetectionOptions(
    val descreenStrength: Float? = null,
    val targetMaxSide: UInt? = null,
    val cropPaddingX: Float? = null,
    val cropPaddingY: Float? = null,
)

/**
 * Phase 2: Detection inference options (limits, normalization).
 */
data class DetectionRunOptions(
    val limitSideLen: Int? = null,
    val limitType: String? = null,
    val mean: FloatArray? = null,
    val std: FloatArray? = null,
)

/**
 * Phase 3: Post-detection options (thresholds, unclip, dilation, box geometry).
 */
data class PostDetectionOptions(
    val threshold: Float? = null,
    val boxThreshold: Float? = null,
    val unclipRatio: Float? = null,
    val useDilation: Boolean? = null,
    val scoreMode: String? = null,
    val maxCandidates: Int? = null,
    val minHeight: Float? = null,
    val minSideLen: Float? = null,
    val maxSideLen: Float? = null,
    val widthHeightRatio: Float? = null,
)

/**
 * Phase 4: Pre-recognition grouping options (line_y, word_x tolerances).
 */
data class PreRecognitionOptions(
    val lineYThreshold: Float? = null,
    val wordXThreshold: Float? = null,
)

/**
 * Phase 5: Recognition inference options (batch size and worker concurrency).
 *
 * @property batchSize Recognition batch size. Default: 10.
 * @property workers Number of parallel worker threads for recognition inference (`null` = auto).
 * @property maxWorkers Maximum worker thread count clamp for recognition inference (`null` = engine default 8).
 */
data class RecognitionRunOptions(
    val batchSize: Int? = null,
    val workers: Int? = null,
    val maxWorkers: Int? = null,
)

/**
 * Phase 6: Post-recognition filtering options (score threshold).
 */
data class PostRecognitionOptions(
    val scoreThreshold: Float? = null,
)

/**
 * Request-local OCR runtime execution options.
 */
data class OcrRunOptions(
    val output: OutputGranularity = OutputGranularity.LINES,
    val autoUnwarp: Boolean? = null,
    val layoutReading: Boolean? = null,
    val layoutModel: String? = null,
    val orientationDetection: Boolean? = null,
    val lineClassification: Boolean? = null,
    val preDetection: PreDetectionOptions? = null,
    val detection: DetectionRunOptions? = null,
    val postDetection: PostDetectionOptions? = null,
    val preRecognition: PreRecognitionOptions? = null,
    val recognition: RecognitionRunOptions? = null,
    val postRecognition: PostRecognitionOptions? = null,
) {
    internal fun toJson(): String {
        return JSONObject().apply {
            put("output", output.wireValue)
            autoUnwarp?.let { put("autoUnwarp", it) }
            layoutReading?.let { put("layoutReading", it) }
            layoutModel?.let { put("layoutModel", it) }
            orientationDetection?.let { put("orientationDetection", it) }
            lineClassification?.let { put("lineClassification", it) }
            preDetection?.let { p ->
                put("preDetection", JSONObject().apply {
                    p.descreenStrength?.let { put("descreenStrength", it.toDouble()) }
                    p.targetMaxSide?.let { put("targetMaxSide", it.toLong()) }
                    p.cropPaddingX?.let { put("cropPaddingX", it.toDouble()) }
                    p.cropPaddingY?.let { put("cropPaddingY", it.toDouble()) }
                })
            }
            detection?.let { d ->
                put("detection", JSONObject().apply {
                    d.limitSideLen?.let { put("limitSideLen", it) }
                    d.limitType?.let { put("limitType", it) }
                    d.mean?.let { put("mean", it.toList()) }
                    d.std?.let { put("std", it.toList()) }
                })
            }
            postDetection?.let { pd ->
                put("postDetection", JSONObject().apply {
                    pd.threshold?.let { put("threshold", it.toDouble()) }
                    pd.boxThreshold?.let { put("boxThreshold", it.toDouble()) }
                    pd.maxCandidates?.let { put("maxCandidates", it) }
                    pd.unclipRatio?.let { put("unclipRatio", it.toDouble()) }
                    pd.useDilation?.let { put("useDilation", it) }
                    pd.scoreMode?.let { put("scoreMode", it) }
                    pd.minHeight?.let { put("minHeight", it.toDouble()) }
                    pd.minSideLen?.let { put("minSideLen", it.toDouble()) }
                    pd.maxSideLen?.let { put("maxSideLen", it.toDouble()) }
                    pd.widthHeightRatio?.let { put("widthHeightRatio", it.toDouble()) }
                })
            }
            preRecognition?.let { pr ->
                put("preRecognition", JSONObject().apply {
                    pr.lineYThreshold?.let { put("lineYThreshold", it.toDouble()) }
                    pr.wordXThreshold?.let { put("wordXThreshold", it.toDouble()) }
                })
            }
            recognition?.let { r ->
                put("recognition", JSONObject().apply {
                    r.batchSize?.let { put("batchSize", it) }
                    r.workers?.let { put("workers", it) }
                    r.maxWorkers?.let { put("maxWorkers", it) }
                })
            }
            postRecognition?.let { pr ->
                put("postRecognition", JSONObject().apply {
                    pr.scoreThreshold?.let { put("scoreThreshold", it.toDouble()) }
                })
            }
        }.toString()
    }
}

/**
 * Universal image source for OCR detection on Android.
 */
sealed interface ImageSource {
    /** File path, file:// URI, or content:// URI. */
    data class Uri(val value: String) : ImageSource
    /** Raw encoded image bytes (JPEG, PNG, etc.). */
    data class Bytes(val value: ByteArray) : ImageSource
}

/**
 * Result of OCR text recognition: structured items or formatted spatial string.
 */
sealed interface DetectTextResult {
    /** Structured list of recognized text lines or words. */
    data class Structured(val items: List<TextResult>) : DetectTextResult
    /** Formatted 2D spatial text layout. */
    data class Spatial(val text: String) : DetectTextResult
}

/**
 * High-performance, pure-Rust OCR engine wrapper for Android.
 */
class RustO private constructor(
    private val context: Context,
    private var engine: uniffi.rusto.RustO?,
) : AutoCloseable {

    companion object {
        /**
         * Downloads model assets for a specific OCR pipeline ahead of time.
         */
        fun downloadModel(model: String? = null, lang: String? = null, cacheDir: String? = null): List<String> =
            uniffi.rusto.downloadModel(model, lang, cacheDir)

        /**
         * Returns the engine library version string.
         */
        fun version(): String = uniffi.rusto.rustoVersion()


        /**
         * Initializes the RustO OCR engine with Android context and optional model configuration.
         *
         * @param context Android context for asset resolution.
         * @param config Model configuration specifying template preset and custom asset paths.
         * @return Initialized RustO engine instance.
         */
        fun initialize(context: Context, config: InitializeConfig = InitializeConfig()): RustO {
            val json = config.toJson(context)
            val engine = try {
                uniffi.rusto.RustO.initializeFromJson(json)
            } catch (e: Exception) {
                throw RustOException("Failed to initialize RustO with config: ${e.message}")
            }
            return RustO(context.applicationContext, engine)
        }

        internal fun copyAssetToCache(context: Context, assetName: String): String {
            val cacheFile = File(context.cacheDir, assetName)
            if (cacheFile.exists() && cacheFile.length() > 0) return cacheFile.absolutePath
            cacheFile.parentFile?.mkdirs()
            val tmpFile = File(cacheFile.parent, "${cacheFile.name}.tmp")
            val candidates = mutableListOf(assetName)
            if (assetName.endsWith(".mnn")) {
                candidates.add(assetName.removeSuffix(".mnn") + ".rten")
                candidates.add(assetName.removeSuffix(".mnn") + ".onnx")
            } else if (assetName.endsWith(".rten")) {
                candidates.add(assetName.removeSuffix(".rten") + ".mnn")
                candidates.add(assetName.removeSuffix(".rten") + ".onnx")
            } else if (assetName.endsWith(".onnx")) {
                candidates.add(assetName.removeSuffix(".onnx") + ".mnn")
                candidates.add(assetName.removeSuffix(".onnx") + ".rten")
            }
            var lastEx: Exception? = null
            for (candidate in candidates) {
                try {
                    context.assets.open(candidate).use { input -> tmpFile.outputStream().use(input::copyTo) }
                    tmpFile.renameTo(cacheFile)
                    return cacheFile.absolutePath
                } catch (e: Exception) {
                    lastEx = e
                    tmpFile.delete()
                }
            }
            throw RustOException("Model asset not found or unreadable: '$assetName'. Cause: ${lastEx?.message}")
        }
    }

    /**
     * Creates a fluent DetectTextBuilder for text detection and recognition.
     */
    fun detectText(): DetectTextBuilder = DetectTextBuilder(this)

    /**
     * Creates a fluent DetectTextBuilder with the given image source.
     */
    fun detectText(source: ImageSource): DetectTextBuilder = DetectTextBuilder(this, source)

    /**
     * Creates a fluent DetectTextBuilder with the given URI string.
     */
    fun detectText(uri: String): DetectTextBuilder = DetectTextBuilder(this, ImageSource.Uri(uri))

    /**
     * Creates a fluent DetectTextBuilder with the given byte array.
     */
    fun detectText(bytes: ByteArray): DetectTextBuilder = DetectTextBuilder(this, ImageSource.Bytes(bytes))

    /**
     * Creates a fluent DetectTextBuilder for text detection and recognition.
     */
    fun detectTextBuilder(): DetectTextBuilder = DetectTextBuilder(this)

    /**
     * Runs text detection and recognition directly with the specified options.
     *
     * @param source ImageSource (Uri or Bytes).
     * @param options Per-request OCR execution options.
     * @return DetectTextResult (Structured or Spatial).
     */
    fun detectText(source: ImageSource, options: OcrRunOptions): DetectTextResult {
        val activeEngine = engine ?: throw IllegalStateException("RustO is closed.")
        val uniffiSource = when (source) {
            is ImageSource.Bytes -> uniffi.rusto.ImageSource.Bytes(source.value)
            is ImageSource.Uri -> withUriInput(source.value) { path, bytes ->
                if (bytes != null) uniffi.rusto.ImageSource.Bytes(bytes)
                else uniffi.rusto.ImageSource.Path(path!!)
            }
        }
        val uniffiOptions = toUniffiOptions(options)

        return try {
            mapDetectTextResult(activeEngine.detectText(uniffiSource, uniffiOptions))
        } catch (e: Exception) {
            throw RustOException("OCR detection failed: ${e.message}")
        }
    }

    /**
     * Runs text detection and recognition directly on a raw interleaved RGB8 pixel buffer.
     */
    fun detectTextRgb(
        pixels: ByteArray,
        width: Int,
        height: Int,
        stride: Long = width.toLong() * 3L,
        options: OcrRunOptions = OcrRunOptions(),
    ): DetectTextResult {
        val activeEngine = engine ?: throw IllegalStateException("RustO is closed.")
        val uniffiOptions = toUniffiOptions(options)
        return try {
            mapDetectTextResult(
                activeEngine.detectTextRgb(
                    pixels,
                    width.toUInt(),
                    height.toUInt(),
                    stride.toULong(),
                    uniffiOptions,
                )
            )
        } catch (e: Exception) {
            throw RustOException("OCR RGB detection failed: ${e.message}")
        }
    }

    private fun mapDetectTextResult(res: uniffi.rusto.DetectTextResult): DetectTextResult =
        when (res) {
            is uniffi.rusto.DetectTextResult.Spatial -> DetectTextResult.Spatial(res.text)
            is uniffi.rusto.DetectTextResult.Structured -> DetectTextResult.Structured(
                res.results.map { item ->
                    TextResult(
                        text = item.text,
                        score = item.score,
                        boxPoints = item.boxPoints.map { Point2D(it.x, it.y) },
                        frame = Frame(
                            width = item.frame.width,
                            height = item.frame.height,
                            top = item.frame.top,
                            left = item.frame.left,
                        ),
                        regionId = item.regionId,
                        layoutType = formatLayoutType(item.layoutType),
                    )
                }
            )
        }

    private fun formatLayoutType(layoutType: uniffi.rusto.LayoutType?): String? =
        when (layoutType) {
            null -> null
            uniffi.rusto.LayoutType.TEXT -> "text"
            uniffi.rusto.LayoutType.TITLE -> "title"
            uniffi.rusto.LayoutType.FIGURE -> "figure"
            uniffi.rusto.LayoutType.FIGURE_CAPTION -> "figureCaption"
            uniffi.rusto.LayoutType.TABLE -> "table"
            uniffi.rusto.LayoutType.TABLE_CAPTION -> "tableCaption"
            uniffi.rusto.LayoutType.HEADER -> "header"
            uniffi.rusto.LayoutType.FOOTER -> "footer"
            uniffi.rusto.LayoutType.REFERENCE -> "reference"
            uniffi.rusto.LayoutType.EQUATION -> "equation"
        }

    private fun toUniffiOptions(options: OcrRunOptions): uniffi.rusto.OcrRunOptions =
        uniffi.rusto.OcrRunOptions(
            output = when (options.output) {
                OutputGranularity.LINES -> uniffi.rusto.OutputGranularity.LINES
                OutputGranularity.WORDS -> uniffi.rusto.OutputGranularity.WORDS
                OutputGranularity.SPATIAL -> uniffi.rusto.OutputGranularity.SPATIAL
                OutputGranularity.TSV -> uniffi.rusto.OutputGranularity.TSV
            },
            autoUnwarp = options.autoUnwarp,
            layoutReading = options.layoutReading,
            layoutModel = options.layoutModel,
            orientationDetection = options.orientationDetection,
            lineClassification = options.lineClassification,
            preDetection = options.preDetection?.let { p ->
                uniffi.rusto.PreDetectionOptions(
                    descreenStrength = p.descreenStrength,
                    targetMaxSide = p.targetMaxSide,
                    cropPaddingX = p.cropPaddingX,
                    cropPaddingY = p.cropPaddingY,
                )
            },
            detection = options.detection?.let { d ->
                uniffi.rusto.DetectionRunOptions(
                    limitSideLen = d.limitSideLen,
                    limitType = d.limitType,
                    mean = d.mean?.toList(),
                    std = d.std?.toList(),
                )
            },
            postDetection = options.postDetection?.let { pd ->
                uniffi.rusto.PostDetectionOptions(
                    threshold = pd.threshold,
                    boxThreshold = pd.boxThreshold,
                    unclipRatio = pd.unclipRatio,
                    useDilation = pd.useDilation,
                    scoreMode = pd.scoreMode,
                    maxCandidates = pd.maxCandidates,
                    minHeight = pd.minHeight,
                    minSideLen = pd.minSideLen,
                    maxSideLen = pd.maxSideLen,
                    widthHeightRatio = pd.widthHeightRatio,
                )
            },
            preRecognition = options.preRecognition?.let { pr ->
                uniffi.rusto.PreRecognitionOptions(
                    lineYThreshold = pr.lineYThreshold,
                    wordXThreshold = pr.wordXThreshold,
                )
            },
            recognition = options.recognition?.let { r ->
                uniffi.rusto.RecognitionRunOptions(
                    batchSize = r.batchSize,
                    workers = r.workers?.toUInt(),
                    maxWorkers = r.maxWorkers?.toUInt(),
                )
            },
            postRecognition = options.postRecognition?.let { pr ->
                uniffi.rusto.PostRecognitionOptions(
                    scoreThreshold = pr.scoreThreshold,
                )
            },
        )

    private inline fun <T> withUriInput(value: String, action: (String?, ByteArray?) -> T): T {
        require(value.isNotBlank()) { "Image URI must be non-empty." }
        if (value.startsWith("content://")) {
            val bytes = context.contentResolver.openInputStream(Uri.parse(value))?.use { it.readBytes() }
                ?: throw RustOException("Failed to open URI: $value")
            return action(null, bytes)
        }
        var path = value.trim()
        if (path.startsWith("file://")) path = Uri.parse(path).path ?: path.removePrefix("file://")
        else if (path.startsWith("file:")) path = path.removePrefix("file:")
        path = try { URLDecoder.decode(path, "UTF-8") } catch (e: Exception) { path }
        require(File(path).isFile) { "Image file not found: $value" }
        return action(path, null)
    }

    /**
     * Checks if the engine instance is initialized and ready for inference.
     */
    fun isReady(): Boolean = engine != null

    /**
     * Returns the engine library version string.
     */
    fun version(): String = uniffi.rusto.rustoVersion()

    /**
     * Releases the native RustO engine handle and frees memory.
     */
    override fun close() {
        engine?.close()
        engine = null
    }
}

/**
 * Fluent builder for chained text detection configuration and execution.
 */
class DetectTextBuilder(
    private val engine: RustO,
    private var source: ImageSource? = null,
) {
    private var rgbPixels: ByteArray? = null
    private var rgbWidth: Int = 0
    private var rgbHeight: Int = 0
    private var rgbStride: Long = 0L
    private var options = OcrRunOptions()

    /**
     * Sets the input image path or URI.
     */
    fun image(uri: String): DetectTextBuilder = apply {
        this.source = ImageSource.Uri(uri)
        this.rgbPixels = null
    }

    /**
     * Sets the input image URI.
     */
    fun image(uri: android.net.Uri): DetectTextBuilder = apply {
        this.source = ImageSource.Uri(uri.toString())
        this.rgbPixels = null
    }

    /**
     * Sets the raw image byte buffer.
     */
    fun data(bytes: ByteArray): DetectTextBuilder = apply {
        this.source = ImageSource.Bytes(bytes)
        this.rgbPixels = null
    }

    /**
     * Sets a raw interleaved RGB8 pixel buffer.
     */
    fun rgb(pixels: ByteArray, width: Int, height: Int, stride: Long = width.toLong() * 3L): DetectTextBuilder = apply {
        this.rgbPixels = pixels
        this.rgbWidth = width
        this.rgbHeight = height
        this.rgbStride = stride
        this.source = null
    }

    /**
     * Sets the image source directly.
     */
    fun source(source: ImageSource): DetectTextBuilder = apply {
        this.source = source
        this.rgbPixels = null
    }

    /**
     * Sets the full OCR execution options.
     */
    fun withOptions(options: OcrRunOptions): DetectTextBuilder = apply {
        this.options = options
    }

    /**
     * Enables or disables automatic document unwarping.
     */
    fun withAutoUnwarp(enabled: Boolean = true): DetectTextBuilder = apply {
        options = options.copy(autoUnwarp = enabled)
    }

    /**
     * Enables or disables document layout reading order analysis.
     */
    fun withLayoutReading(enabled: Boolean = true): DetectTextBuilder = apply {
        options = options.copy(layoutReading = enabled)
    }

    /**
     * Sets document layout or reading direction model identifier.
     */
    fun withLayoutModel(model: String): DetectTextBuilder = apply {
        options = options.copy(layoutModel = model)
    }

    /**
     * Enables or disables 0/90/180/270 orientation angle detection.
     */
    fun withOrientationDetection(enabled: Boolean = true): DetectTextBuilder = apply {
        options = options.copy(orientationDetection = enabled)
    }

    /**
     * Enables or disables text line classification (e.g. 180-degree flip).
     */
    fun withLineClassification(enabled: Boolean = true): DetectTextBuilder = apply {
        options = options.copy(lineClassification = enabled)
    }

    /**
     * Configures pre-detection options.
     */
    fun withPreDetection(preDetection: PreDetectionOptions): DetectTextBuilder = apply {
        options = options.copy(preDetection = preDetection)
    }

    /**
     * Configures pre-detection options via trailing lambda.
     */
    fun withPreDetection(configure: PreDetectionOptions.() -> PreDetectionOptions): DetectTextBuilder = apply {
        val current = options.preDetection ?: PreDetectionOptions()
        options = options.copy(preDetection = current.configure())
    }

    /**
     * Configures detection inference options.
     */
    fun withDetection(detection: DetectionRunOptions): DetectTextBuilder = apply {
        options = options.copy(detection = detection)
    }

    /**
     * Configures detection inference options via trailing lambda.
     */
    fun withDetection(configure: DetectionRunOptions.() -> DetectionRunOptions): DetectTextBuilder = apply {
        val current = options.detection ?: DetectionRunOptions()
        options = options.copy(detection = current.configure())
    }

    /**
     * Configures post-detection threshold and filtering options.
     */
    fun withPostDetection(postDetection: PostDetectionOptions): DetectTextBuilder = apply {
        options = options.copy(postDetection = postDetection)
    }

    /**
     * Configures post-detection options via trailing lambda.
     */
    fun withPostDetection(configure: PostDetectionOptions.() -> PostDetectionOptions): DetectTextBuilder = apply {
        val current = options.postDetection ?: PostDetectionOptions()
        options = options.copy(postDetection = current.configure())
    }

    /**
     * Configures pre-recognition line/word grouping tolerances.
     */
    fun withPreRecognition(preRecognition: PreRecognitionOptions): DetectTextBuilder = apply {
        options = options.copy(preRecognition = preRecognition)
    }

    /**
     * Configures pre-recognition options via trailing lambda.
     */
    fun withPreRecognition(configure: PreRecognitionOptions.() -> PreRecognitionOptions): DetectTextBuilder = apply {
        val current = options.preRecognition ?: PreRecognitionOptions()
        options = options.copy(preRecognition = current.configure())
    }

    /**
     * Configures recognition execution options.
     */
    fun withRecognition(recognition: RecognitionRunOptions): DetectTextBuilder = apply {
        options = options.copy(recognition = recognition)
    }

    /**
     * Configures recognition execution options via trailing lambda.
     */
    fun withRecognition(configure: RecognitionRunOptions.() -> RecognitionRunOptions): DetectTextBuilder = apply {
        val current = options.recognition ?: RecognitionRunOptions()
        options = options.copy(recognition = current.configure())
    }

    /**
     * Configures post-recognition confidence filtering options.
     */
    fun withPostRecognition(postRecognition: PostRecognitionOptions): DetectTextBuilder = apply {
        options = options.copy(postRecognition = postRecognition)
    }

    /**
     * Configures post-recognition options via trailing lambda.
     */
    fun withPostRecognition(configure: PostRecognitionOptions.() -> PostRecognitionOptions): DetectTextBuilder = apply {
        val current = options.postRecognition ?: PostRecognitionOptions()
        options = options.copy(postRecognition = current.configure())
    }

    /**
     * Sets output format granularity.
     */
    fun withOutput(granularity: OutputGranularity): DetectTextBuilder = apply {
        options = options.copy(output = granularity)
    }

    /**
     * Sets output granularity to lines.
     */
    fun asLines(): DetectTextBuilder = withOutput(OutputGranularity.LINES)

    /**
     * Sets output granularity to individual words.
     */
    fun asWords(): DetectTextBuilder = withOutput(OutputGranularity.WORDS)

    /**
     * Sets output granularity to 2D formatted spatial text.
     */
    fun asSpatial(): DetectTextBuilder = withOutput(OutputGranularity.SPATIAL)

    /**
     * Sets output granularity to TSV (tab-separated values).
     */
    fun asTsv(): DetectTextBuilder = withOutput(OutputGranularity.TSV)

    /**
     * Terminal execution: runs text detection and returns the result.
     */
    fun run(): DetectTextResult {
        val pixels = rgbPixels
        if (pixels != null) {
            return engine.detectTextRgb(pixels, rgbWidth, rgbHeight, rgbStride, options)
        }
        val src = source ?: throw IllegalStateException("Image source must be specified before calling run().")
        return engine.detectText(src, options)
    }
}


