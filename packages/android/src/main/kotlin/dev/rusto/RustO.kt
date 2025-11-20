package dev.rusto

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.ByteArrayOutputStream

data class Point2D(val x: Float, val y: Float)

data class TextResult(
    val text: String,
    val score: Float,
    val boxPoints: List<Point2D>
)

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
        private external fun nativeFreeResults(resultsPtr: Long, count: Int)

        @JvmStatic
        private external fun nativeFree(handle: Long)

        fun create(
            context: Context,
            detModel: String,
            recModel: String,
            dict: String
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

        private fun copyAssetToCache(context: Context, assetName: String): String {
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
        val resultsOut = LongArray(1)
        val countOut = IntArray(1)

        val status = nativeOcrFile(nativeHandle, imagePath, resultsOut, countOut)
        if (status != 0) {
            throw RustOException("OCR recognition failed with status: $status")
        }

        try {
            return parseResults(resultsOut[0], countOut[0])
        } finally {
            nativeFreeResults(resultsOut[0], countOut[0])
        }
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

    private fun parseResults(resultsPtr: Long, count: Int): List<TextResult> {
        val results = mutableListOf<TextResult>()
        
        for (i in 0 until count) {
            val textOut = Array(1) { "" }
            val scoreOut = FloatArray(1)
            val boxOut = FloatArray(8)

            nativeGetResult(resultsPtr, i, textOut, scoreOut, boxOut)

            results.add(
                TextResult(
                    text = textOut[0],
                    score = scoreOut[0],
                    boxPoints = listOf(
                        Point2D(boxOut[0], boxOut[1]),
                        Point2D(boxOut[2], boxOut[3]),
                        Point2D(boxOut[4], boxOut[5]),
                        Point2D(boxOut[6], boxOut[7])
                    )
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
