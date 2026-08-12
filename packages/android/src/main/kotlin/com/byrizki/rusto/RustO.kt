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

data class RustOConfig(
    val template: String = "ppv6",
    val detModelPath: String = "det.mnn",
    val recModelPath: String = "rec.mnn",
    val dictPath: String = "dict.txt",
    val clsModelPath: String? = null,
    val orientModelPath: String? = null,
    val unwarpModelPath: String? = null,
    val orientThreshold: Float? = null,
    val clsThreshold: Float? = null,
    val textScore: Float = 0.5f,
    val detThresh: Float = 0.3f,
    val detBoxThresh: Float = 0.6f,
    val limitSideLen: Int = 736,
    val limitType: String = "min",
    val unclipRatio: Float = 2.0f,
    val useDilation: Boolean = true,
    val useDet: Boolean = true,
    val useRec: Boolean = true,
    val useCls: Boolean = false,
    val useOrient: Boolean = false,
    val useUnwarp: Boolean = false,
    val debugImages: Boolean = false,
    val minHeight: Float = 30.0f,
    val maxSideLen: Float = 2000.0f,
    val minSideLen: Float = 30.0f,
    val returnWordBox: Boolean = false,
    val returnSingleCharBox: Boolean = false,
    val yThresholdMultiplier: Float? = null,
    val xThresholdMultiplier: Float? = null
) {
    fun toJson(context: Context): String {
        val json = JSONObject()
        json.put("template", template)
        json.put("detModelPath", resolveModel(context, detModelPath))
        json.put("recModelPath", resolveModel(context, recModelPath))
        json.put("dictPath", resolveModel(context, dictPath))
        clsModelPath?.let { json.put("clsModelPath", resolveModel(context, it)) }
        orientModelPath?.let { json.put("orientModelPath", resolveModel(context, it)) }
        unwarpModelPath?.let { json.put("unwarpModelPath", resolveModel(context, it)) }
        orientThreshold?.let { json.put("orientThreshold", it.toDouble()) }
        clsThreshold?.let { json.put("clsThreshold", it.toDouble()) }
        json.put("textScore", textScore.toDouble())
        json.put("detThresh", detThresh.toDouble())
        json.put("detBoxThresh", detBoxThresh.toDouble())
        json.put("limitSideLen", limitSideLen)
        json.put("limitType", limitType)
        json.put("unclipRatio", unclipRatio.toDouble())
        json.put("useDilation", useDilation)
        json.put("useDet", useDet)
        json.put("useRec", useRec)
        json.put("useCls", useCls)
        json.put("useOrient", useOrient)
        json.put("useUnwarp", useUnwarp)
        json.put("debugImages", debugImages)
        json.put("minHeight", minHeight.toDouble())
        json.put("maxSideLen", maxSideLen.toDouble())
        json.put("minSideLen", minSideLen.toDouble())
        json.put("returnWordBox", returnWordBox)
        json.put("returnSingleCharBox", returnSingleCharBox)
        yThresholdMultiplier?.let { json.put("yThresholdMultiplier", it.toDouble()) }
        xThresholdMultiplier?.let { json.put("xThresholdMultiplier", it.toDouble()) }
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
        private external fun nativeNew(
            detModelPath: String,
            recModelPath: String,
            dictPath: String
        ): Long

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
            config: RustOConfig
        ): RustO {
            val configJson = config.toJson(context)
            val handle = nativeNewWithConfig(configJson)
            if (handle == 0L) {
                throw RustOException("Failed to initialize RustO with config")
            }
            return RustO(handle)
        }

        fun create(
            context: Context,
            detModel: String = "det.mnn",
            recModel: String = "rec.mnn",
            dict: String = "dict.txt"
        ): RustO {
            val detPath = copyAssetToCache(context, detModel)
            val recPath = copyAssetToCache(context, recModel)
            val dictPath = copyAssetToCache(context, dict)

            val handle = nativeNew(detPath, recPath, dictPath)
            if (handle == 0L) {
                throw RustOException("Failed to initialize RustO")
            }
            return RustO(handle)
        }

        internal fun copyAssetToCache(context: Context, assetName: String): String {
            val cacheFile = File(context.cacheDir, assetName)
            if (!cacheFile.exists()) {
                cacheFile.parentFile?.mkdirs()
                context.assets.open(assetName).use { input ->
                    cacheFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
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
