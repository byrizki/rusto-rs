package com.byrizki.rusto

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import org.json.JSONObject
import java.io.ByteArrayOutputStream
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

data class PreprocessingConfig(
    val minHeight: Float? = null,
    val maxSideLen: Float? = null,
    val minSideLen: Float? = null,
    val debugImages: Boolean? = null
)

data class LayoutConfig(
    val yThresholdMultiplier: Float? = null,
    val xThresholdMultiplier: Float? = null
)

data class RustOConfig(
    val template: String? = null,
    val detection: DetectionConfig? = null,
    val recognition: RecognitionConfig? = null,
    val classification: ClassificationConfig? = null,
    val orientation: OrientationConfig? = null,
    val unwarp: UnwarpConfig? = null,
    val preprocessing: PreprocessingConfig? = null,
    val layout: LayoutConfig? = null
) {
    fun toJson(context: Context): String {
        val json = JSONObject()
        template?.let { json.put("template", it) }

        val detObj = JSONObject()
        val detPath = detection?.modelPath ?: "det.mnn"
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
        val recPath = recognition?.modelPath ?: "rec.mnn"
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

        preprocessing?.let { prep ->
            val prepObj = JSONObject()
            prep.minHeight?.let { prepObj.put("minHeight", it.toDouble()) }
            prep.maxSideLen?.let { prepObj.put("maxSideLen", it.toDouble()) }
            prep.minSideLen?.let { prepObj.put("minSideLen", it.toDouble()) }
            prep.debugImages?.let { prepObj.put("debugImages", it) }
            if (prepObj.length() > 0) json.put("preprocessing", prepObj)
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

class RustO private constructor(
    private val nativeHandle: Long
) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("rusto")
        }

        @JvmStatic
        external fun nativeVersion(): String

        @JvmStatic
        private external fun nativeNewWithConfig(
            configJson: String
        ): Long

        @JvmStatic
        private external fun nativeOcrFile(
            handle: Long,
            imagePath: String,
            resultsOut: LongArray,
            countOut: IntArray
        ): Int

        @JvmStatic
        private external fun nativeOcrData(
            handle: Long,
            imageData: ByteArray,
            resultsOut: LongArray,
            countOut: IntArray
        ): Int

        @JvmStatic
        private external fun nativeGetResult(
            resultsPtr: Long,
            index: Int,
            textOut: Array<String>,
            scoreOut: FloatArray,
            boxOut: FloatArray
        )

        @JvmStatic
        private external fun nativeOcrFileWithOutput(
            handle: Long,
            imagePath: String
        ): Long

        @JvmStatic
        private external fun nativeOcrDataWithOutput(
            handle: Long,
            imageData: ByteArray
        ): Long

        @JvmStatic
        private external fun nativeOutputToRaw(
            output: Long
        ): String?

        @JvmStatic
        private external fun nativeOutputToCsv(
            output: Long
        ): String?

        @JvmStatic
        private external fun nativeOutputToTextWithPosition(
            output: Long
        ): String?

        @JvmStatic
        private external fun nativeOutputToSpatialText(
            output: Long,
            yThresholdMultiplier: Float,
            xThresholdMultiplier: Float
        ): String?

        @JvmStatic
        private external fun nativeFreeOutput(
            output: Long
        )

        @JvmStatic
        private external fun nativeFreeResults(resultsPtr: Long, count: Int)

        @JvmStatic
        private external fun nativeFree(handle: Long)

        fun create(
            context: Context,
            config: RustOConfig = RustOConfig()
        ): RustO {
            val configJson = config.toJson(context)
            val handle = nativeNewWithConfig(configJson)
            if (handle == 0L) {
                throw RustOException("Failed to initialize RustO with config")
            }
            return RustO(handle)
        }

        internal fun copyAssetToCache(context: Context, assetName: String): String {
            val cacheFile = File(context.cacheDir, assetName)
            if (cacheFile.exists() && cacheFile.length() > 0) return cacheFile.absolutePath

            cacheFile.parentFile?.mkdirs()
            val tmpFile = File(cacheFile.parent, "${cacheFile.name}.tmp")
            try {
                context.assets.open(assetName).use { input ->
                    tmpFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                tmpFile.renameTo(cacheFile)
            } catch (e: java.io.IOException) {
                tmpFile.delete()
                throw RustOException("Model asset not found or unreadable: '$assetName'. " +
                    "Ensure the file is bundled in assets or provide an absolute path. Cause: ${e.message}")
            }
            return cacheFile.absolutePath
        }
    }

    val version: String
        get() = nativeVersion()

    fun recognizeFile(imagePath: String): List<TextResult> {
        val cleanPath = cleanPath(imagePath)
        val resultsOut = LongArray(1)
        val countOut = IntArray(1)

        val status = nativeOcrFile(nativeHandle, cleanPath, resultsOut, countOut)
        if (status != 0) {
            throw RustOException("OCR recognition failed with status: $status")
        }

        try {
            return parseResults(resultsOut[0], countOut[0])
        } finally {
            nativeFreeResults(resultsOut[0], countOut[0])
        }
    }

    fun recognize(context: Context, uri: Uri): List<TextResult> {
        if (uri.scheme == "content") {
            val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                ?: throw RustOException("Failed to open URI: $uri")
            return recognize(bytes)
        }
        val path = uri.path ?: uri.toString()
        return recognizeFile(path)
    }

    fun recognize(imageData: ByteArray): List<TextResult> {
        val resultsOut = LongArray(1)
        val countOut = IntArray(1)

        val status = nativeOcrData(nativeHandle, imageData, resultsOut, countOut)
        if (status != 0) {
            throw RustOException("OCR recognition failed with status: $status")
        }

        try {
            return parseResults(resultsOut[0], countOut[0])
        } finally {
            nativeFreeResults(resultsOut[0], countOut[0])
        }
    }

    fun recognize(bitmap: Bitmap): List<TextResult> {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
        return recognize(stream.toByteArray())
    }

    fun recognizeFileToRaw(imagePath: String): String {
        return runWithOutput(cleanPath(imagePath)) { nativeOutputToRaw(it) ?: "" }
    }

    fun recognizeFileToCsv(imagePath: String): String {
        return runWithOutput(cleanPath(imagePath)) { nativeOutputToCsv(it) ?: "" }
    }

    fun recognizeFileToTextWithPosition(imagePath: String): String {
        return runWithOutput(cleanPath(imagePath)) { nativeOutputToTextWithPosition(it) ?: "" }
    }

    fun recognizeFileToSpatialText(
        imagePath: String,
        yThresholdMultiplier: Float = 0f,
        xThresholdMultiplier: Float = 0f
    ): String {
        return runWithOutput(cleanPath(imagePath)) {
            nativeOutputToSpatialText(it, yThresholdMultiplier, xThresholdMultiplier) ?: ""
        }
    }

    fun recognizeToRaw(imageData: ByteArray): String {
        return runWithOutput(imageData) { nativeOutputToRaw(it) ?: "" }
    }

    fun recognizeToCsv(imageData: ByteArray): String {
        return runWithOutput(imageData) { nativeOutputToCsv(it) ?: "" }
    }

    fun recognizeToTextWithPosition(imageData: ByteArray): String {
        return runWithOutput(imageData) { nativeOutputToTextWithPosition(it) ?: "" }
    }

    fun recognizeToSpatialText(
        imageData: ByteArray,
        yThresholdMultiplier: Float = 0f,
        xThresholdMultiplier: Float = 0f
    ): String {
        return runWithOutput(imageData) {
            nativeOutputToSpatialText(it, yThresholdMultiplier, xThresholdMultiplier) ?: ""
        }
    }

    private fun cleanPath(imagePath: String): String {
        var clean = imagePath.trim()
        if (clean.startsWith("file://")) {
            val uri = Uri.parse(clean)
            clean = uri.path ?: clean.removePrefix("file://")
        } else if (clean.startsWith("file:")) {
            clean = clean.removePrefix("file:")
        }
        try {
            clean = URLDecoder.decode(clean, "UTF-8")
        } catch (_: Exception) {}
        return clean
    }

    private inline fun runWithOutput(imagePath: String, action: (Long) -> String): String {
        val output = nativeOcrFileWithOutput(nativeHandle, imagePath)
        if (output == 0L) {
            throw RustOException("OCR recognition failed")
        }
        try {
            return action(output)
        } finally {
            nativeFreeOutput(output)
        }
    }

    private inline fun runWithOutput(imageData: ByteArray, action: (Long) -> String): String {
        val output = nativeOcrDataWithOutput(nativeHandle, imageData)
        if (output == 0L) {
            throw RustOException("OCR recognition failed")
        }
        try {
            return action(output)
        } finally {
            nativeFreeOutput(output)
        }
    }

    private fun parseResults(resultsPtr: Long, count: Int): List<TextResult> {
        val results = ArrayList<TextResult>(count)

        val textOut = arrayOf("")
        val scoreOut = FloatArray(1)
        val boxOut = FloatArray(8)

        for (i in 0 until count) {
            nativeGetResult(resultsPtr, i, textOut, scoreOut, boxOut)

            val boxPoints = listOf(
                Point2D(boxOut[0], boxOut[1]),
                Point2D(boxOut[2], boxOut[3]),
                Point2D(boxOut[4], boxOut[5]),
                Point2D(boxOut[6], boxOut[7])
            )

            val minX = minOf(boxOut[0], boxOut[2], boxOut[4], boxOut[6])
            val maxX = maxOf(boxOut[0], boxOut[2], boxOut[4], boxOut[6])
            val minY = minOf(boxOut[1], boxOut[3], boxOut[5], boxOut[7])
            val maxY = maxOf(boxOut[1], boxOut[3], boxOut[5], boxOut[7])

            val frame = Frame(
                width = maxX - minX,
                height = maxY - minY,
                top = minY,
                left = minX
            )

            results.add(
                TextResult(
                    text = textOut[0],
                    score = scoreOut[0],
                    boxPoints = boxPoints,
                    frame = frame
                )
            )
        }

        return results
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeFree(nativeHandle)
        }
    }
}
