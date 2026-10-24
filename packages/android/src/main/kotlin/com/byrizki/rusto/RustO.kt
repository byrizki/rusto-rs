package com.byrizki.rusto

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.net.URLDecoder

data class Point2D(val x: Float, val y: Float)

data class Frame(
    val width: Float,
    val height: Float,
    val top: Float,
    val left: Float
)

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
    }
)

data class DetectionConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null,
    val thresh: Float? = null,
    val boxThresh: Float? = null,
    val unclipRatio: Float? = null,
    val limitSideLen: Int? = null,
    val limitType: String? = null,
    val useDilation: Boolean? = null
)

data class RecognitionConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null,
    val dictPath: String? = null,
    val scoreThresh: Float? = null,
    val returnWordBox: Boolean? = null,
    val returnSingleCharBox: Boolean? = null
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

data class OrientationConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null,
    val thresh: Float? = null
)

data class UnwarpConfig(
    val enabled: Boolean? = null,
    val modelPath: String? = null
)

data class LayoutConfig(
    val yThresholdMultiplier: Float? = null,
    val xThresholdMultiplier: Float? = null
)

data class InitializeConfig(
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
        template?.let { json.put("template", it) }

        val detObj = JSONObject()
        val detPath = detection?.modelPath ?: "det.rten"
        detObj.put("modelPath", resolveModel(context, detPath))
        detection?.enabled?.let { detObj.put("enabled", it) }
        detection?.thresh?.let { detObj.put("thresh", it.toDouble()) }
        detection?.boxThresh?.let { detObj.put("boxThresh", it.toDouble()) }
        detection?.unclipRatio?.let { detObj.put("unclipRatio", it.toDouble()) }
        detection?.limitSideLen?.let { detObj.put("limitSideLen", it) }
        detection?.limitType?.let { detObj.put("limitType", it) }
        detection?.useDilation?.let { detObj.put("useDilation", it) }
        json.put("detection", detObj)

        val recObj = JSONObject()
        val recPath = recognition?.modelPath ?: "rec.rten"
        val dictPath = recognition?.dictPath ?: "dict.txt"
        recObj.put("modelPath", resolveModel(context, recPath))
        recObj.put("dictPath", resolveModel(context, dictPath))
        recognition?.enabled?.let { recObj.put("enabled", it) }
        recognition?.scoreThresh?.let { recObj.put("scoreThresh", it.toDouble()) }
        recognition?.returnWordBox?.let { recObj.put("returnWordBox", it) }
        recognition?.returnSingleCharBox?.let { recObj.put("returnSingleCharBox", it) }
        json.put("recognition", recObj)

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
            lay.yThresholdMultiplier?.let { layObj.put("yThresholdMultiplier", it.toDouble()) }
            lay.xThresholdMultiplier?.let { layObj.put("xThresholdMultiplier", it.toDouble()) }
            if (layObj.length() > 0) json.put("layout", layObj)
        }

        return json.toString()
    }

    private fun resolveModel(context: Context, pathOrAsset: String): String {
        val file = File(pathOrAsset)
        if (file.exists()) return file.absolutePath
        return RustO.copyAssetToCache(context, pathOrAsset)
    }
}

class RustOException(message: String) : Exception(message)

enum class OutputGranularity(val wireValue: String) {
    LINES("lines"), WORDS("words"), SPATIAL("spatial");
}

data class PostprocessRunOptions(
    val threshold: Float? = null,
    val boxThreshold: Float? = null,
    val maxCandidates: Int? = null,
    val unclipRatio: Float? = null,
    val useDilation: Boolean? = null,
)

data class DetectionRunOptions(
    val limitSideLen: Int? = null,
    val limitType: String? = null,
    val mean: FloatArray? = null,
    val std: FloatArray? = null,
    val postprocess: PostprocessRunOptions? = null,
)

data class OcrRunOptions(
    val output: OutputGranularity = OutputGranularity.LINES,
    val lineYThreshold: Float? = null,
    val wordXThreshold: Float? = null,
    val textScore: Float? = null,
    val classification: Boolean? = null,
    val orientation: Boolean? = null,
    val minHeight: Float? = null,
    val maxSideLen: Float? = null,
    val minSideLen: Float? = null,
    val widthHeightRatio: Float? = null,
    val detection: DetectionRunOptions? = null,
    val postprocess: PostprocessRunOptions? = null,
) {
    internal fun toJson(): String {
        require(lineYThreshold == null || lineYThreshold.isFinite() && lineYThreshold >= 0f) {
            "lineYThreshold must be finite and non-negative."
        }
        require(wordXThreshold == null || wordXThreshold.isFinite() && wordXThreshold >= 0f) {
            "wordXThreshold must be finite and non-negative."
        }
        require(textScore == null || textScore.isFinite() && textScore in 0f..1f) {
            "textScore must be finite and between 0 and 1."
        }
        validateRuntimeOverrides()
        return JSONObject().apply {
            put("output", output.wireValue)
            lineYThreshold?.let { put("lineYThreshold", it.toDouble()) }
            wordXThreshold?.let { put("wordXThreshold", it.toDouble()) }
            textScore?.let { put("textScore", it.toDouble()) }
            classification?.let { put("classification", it) }
            orientation?.let { put("orientation", it) }
            minHeight?.let { put("minHeight", it.toDouble()) }
            maxSideLen?.let { put("maxSideLen", it.toDouble()) }
            minSideLen?.let { put("minSideLen", it.toDouble()) }
            widthHeightRatio?.let { put("widthHeightRatio", it.toDouble()) }
            detection?.let { detection ->
                put("detection", JSONObject().apply {
                    detection.limitSideLen?.let { put("limitSideLen", it) }
                    detection.limitType?.let { put("limitType", it) }
                    detection.mean?.let { put("mean", it.toList()) }
                    detection.std?.let { put("std", it.toList()) }
                })
            }
            postprocess?.let { postprocess ->
                put("postprocess", JSONObject().apply {
                    postprocess.threshold?.let { put("threshold", it.toDouble()) }
                    postprocess.boxThreshold?.let { put("boxThreshold", it.toDouble()) }
                    postprocess.maxCandidates?.let { put("maxCandidates", it) }
                    postprocess.unclipRatio?.let { put("unclipRatio", it.toDouble()) }
                    postprocess.useDilation?.let { put("useDilation", it) }
                })
            }
        }.toString()
    }

    private fun validateRuntimeOverrides() {
        fun positive(name: String, number: Float?) = require(number == null || number.isFinite() && number > 0f) { "$name must be finite and greater than zero." }
        positive("minHeight", minHeight); positive("maxSideLen", maxSideLen); positive("minSideLen", minSideLen)
        require(widthHeightRatio == null || widthHeightRatio.isFinite() && (widthHeightRatio > 0f || widthHeightRatio == -1f)) { "widthHeightRatio must be positive or -1." }
        require(minSideLen == null || maxSideLen == null || minSideLen <= maxSideLen) { "minSideLen must not exceed maxSideLen." }
        detection?.let { detection ->
            require(detection.limitSideLen == null || detection.limitSideLen in 1..32767) { "limitSideLen must be between 1 and 32767." }
            require(detection.limitType == null || detection.limitType == "min" || detection.limitType == "max") { "limitType must be min or max." }
            for ((name, values, rejectZero) in listOf(Triple("mean", detection.mean, false), Triple("std", detection.std, true))) {
                require(values == null || values.size == 3 && values.all { it.isFinite() && (!rejectZero || it != 0f) }) { "$name must contain three valid numbers." }
            }
        }
        postprocess?.let { postprocess ->
            require(postprocess.threshold == null || postprocess.threshold.isFinite() && postprocess.threshold in 0f..1f) { "threshold must be between zero and one." }
            require(postprocess.boxThreshold == null || postprocess.boxThreshold.isFinite() && postprocess.boxThreshold in 0f..1f) { "boxThreshold must be between zero and one." }
            require(postprocess.maxCandidates == null || postprocess.maxCandidates >= 1) { "maxCandidates must be at least one." }
            require(postprocess.unclipRatio == null || postprocess.unclipRatio.isFinite() && postprocess.unclipRatio > 0f) { "unclipRatio must be finite and greater than zero." }
        }
    }
}

sealed interface ImageSource {
    data class Uri(val value: String) : ImageSource
    data class Bytes(val value: ByteArray) : ImageSource
}

sealed interface DetectTextResult {
    data class Structured(val items: List<TextResult>) : DetectTextResult
    data class Spatial(val text: String) : DetectTextResult
}

class RustO private constructor(
    private val context: Context,
    private var engine: uniffi.rusto.RustO?,
) : AutoCloseable {

    companion object {
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
            if (assetName.endsWith(".rten")) {
                candidates.add(assetName.removeSuffix(".rten") + ".onnx")
            } else if (assetName.endsWith(".onnx")) {
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

    fun detectText(source: ImageSource, options: OcrRunOptions = OcrRunOptions()): DetectTextResult {
        val activeEngine = engine ?: throw IllegalStateException("RustO is closed.")
        val uniffiSource = when (source) {
            is ImageSource.Bytes -> uniffi.rusto.ImageSource.Bytes(source.value)
            is ImageSource.Uri -> withUriInput(source.value) { path, bytes ->
                if (bytes != null) uniffi.rusto.ImageSource.Bytes(bytes)
                else uniffi.rusto.ImageSource.Path(path!!)
            }
        }
        val uniffiOptions = uniffi.rusto.OcrRunOptions(
            output = when (options.output) {
                OutputGranularity.LINES -> uniffi.rusto.OutputGranularity.LINES
                OutputGranularity.WORDS -> uniffi.rusto.OutputGranularity.WORDS
                OutputGranularity.SPATIAL -> uniffi.rusto.OutputGranularity.SPATIAL
            },
            lineYThreshold = options.lineYThreshold,
            wordXThreshold = options.wordXThreshold,
            textScore = options.textScore,
            classification = options.classification,
            orientation = options.orientation,
            minHeight = options.minHeight,
            maxSideLen = options.maxSideLen,
            minSideLen = options.minSideLen,
            widthHeightRatio = options.widthHeightRatio,
            detection = options.detection?.let { d ->
                uniffi.rusto.DetectionRunOptions(
                    limitSideLen = d.limitSideLen,
                    limitType = d.limitType,
                    mean = d.mean?.toList(),
                    std = d.std?.toList(),
                    postprocess = d.postprocess?.let { p ->
                        uniffi.rusto.PostprocessRunOptions(
                            threshold = p.threshold,
                            boxThreshold = p.boxThreshold,
                            maxCandidates = p.maxCandidates,
                            unclipRatio = p.unclipRatio,
                            useDilation = p.useDilation,
                        )
                    }
                )
            },
            postprocess = options.postprocess?.let { p ->
                uniffi.rusto.PostprocessRunOptions(
                    threshold = p.threshold,
                    boxThreshold = p.boxThreshold,
                    maxCandidates = p.maxCandidates,
                    unclipRatio = p.unclipRatio,
                    useDilation = p.useDilation,
                )
            },
            calibration = null,
            optimization = null,
        )

        return try {
            when (val res = activeEngine.detectText(uniffiSource, uniffiOptions)) {
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
                            )
                        )
                    }
                )
            }
        } catch (e: Exception) {
            throw RustOException("OCR detection failed: ${e.message}")
        }
    }

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

    override fun close() {
        engine?.close()
        engine = null
    }
}

