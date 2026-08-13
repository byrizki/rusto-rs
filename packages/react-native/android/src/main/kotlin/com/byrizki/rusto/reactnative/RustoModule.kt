package com.byrizki.rusto.reactnative

import android.util.Base64
import com.byrizki.rusto.ClassificationConfig
import com.byrizki.rusto.DetectionConfig
import com.byrizki.rusto.ImageSource
import com.byrizki.rusto.OcrRunOptions
import com.byrizki.rusto.OutputGranularity
import com.byrizki.rusto.DetectTextResult
import com.byrizki.rusto.OrientationConfig
import com.byrizki.rusto.RecognitionConfig
import com.byrizki.rusto.RustO
import com.byrizki.rusto.InitializeConfig
import com.byrizki.rusto.TextResult
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableArray

class RustoModule(private val reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
    private var rustoInstance: RustO? = null
    override fun getName() = "Rusto"

    @ReactMethod
    fun initialize(configMap: ReadableMap?, promise: Promise) {
        try {
            rustoInstance?.close()
            rustoInstance = RustO.initialize(reactContext, parseInitializeConfig(configMap))
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", "Failed to initialize RustO: ${e.message}", e)
        }
    }

    private fun parseInitializeConfig(map: ReadableMap?): InitializeConfig {
        if (map == null) return InitializeConfig()
        val models = if (map.hasKey("models") && !map.isNull("models")) map.getMap("models") else null
        fun model(name: String): String? = models?.takeIf { it.hasKey(name) && !it.isNull(name) }?.getString(name)
        return InitializeConfig(
            template = if (map.hasKey("preset") && !map.isNull("preset")) map.getString("preset") else "ppv6",
            detection = DetectionConfig(modelPath = model("detection") ?: "det.mnn"),
            recognition = RecognitionConfig(modelPath = model("recognition") ?: "rec.mnn", dictPath = model("dictionary") ?: "dict.txt"),
            classification = model("classification")?.let { ClassificationConfig(enabled = true, modelPath = it) },
            orientation = model("orientation")?.let { OrientationConfig(enabled = true, modelPath = it) },
        )
    }

    @ReactMethod
    fun detectText(source: ReadableMap, options: ReadableMap?, promise: Promise) {
        try {
            val instance = rustoInstance ?: throw IllegalStateException("RustO not initialized. Call initialize() first.")
            when (val result = instance.detectText(parseSource(source), parseOptions(options))) {
                is DetectTextResult.Spatial -> promise.resolve(result.text)
                is DetectTextResult.Structured -> promise.resolve(toWritableArray(result.items))
            }
        } catch (e: IllegalStateException) {
            promise.reject("NOT_INITIALIZED", e.message, e)
        } catch (e: IllegalArgumentException) {
            val code = if (e.message?.startsWith("Provide exactly one") == true || e.message?.startsWith("Source value") == true) "INVALID_SOURCE" else "INVALID_OPTIONS"
            promise.reject(code, e.message, e)
        } catch (e: Exception) {
            promise.reject("OCR_ERROR", "Failed to detect text: ${e.message}", e)
        }
    }

    private fun parseSource(source: ReadableMap): ImageSource {
        val allowedKeys = setOf("uri", "base64")
        val iterator = source.keySetIterator()
        val keys = mutableListOf<String>()
        while (iterator.hasNextKey()) keys += iterator.nextKey()
        require(keys.all { it in allowedKeys } && keys.size == 1) {
            "Provide exactly one source key: uri or base64."
        }
        val sourceKey = keys.single()
        require(!source.isNull(sourceKey) && !source.getString(sourceKey).isNullOrBlank()) {
            "Source value must be non-empty."
        }
        return when (sourceKey) {
            "base64" -> ImageSource.Bytes(decodeBase64(source.getString("base64")!!))
            else -> ImageSource.Uri(source.getString("uri")!!)
        }
    }

    private fun parseOptions(options: ReadableMap?): OcrRunOptions {
        val output = options?.takeIf { it.hasKey("output") && !it.isNull("output") }?.getString("output") ?: "lines"
        require(output in setOf("lines", "words", "spatial")) { "Invalid output: $output" }
        fun number(name: String): Double? = options?.takeIf { it.hasKey(name) && !it.isNull(name) }?.getDouble(name)
        val y = number("lineYThreshold")
        val x = number("wordXThreshold")
        val score = number("textScore")
        require((y == null || y.isFinite() && y >= 0) && (x == null || x.isFinite() && x >= 0) && (score == null || score.isFinite() && score in 0.0..1.0)) { "Invalid runtime options" }
        return OcrRunOptions(
            output = OutputGranularity.entries.first { it.wireValue == output },
            lineYThreshold = y?.toFloat(),
            wordXThreshold = x?.toFloat(),
            textScore = score?.toFloat(),
            classification = options?.takeIf { it.hasKey("classification") && !it.isNull("classification") }?.getBoolean("classification"),
            orientation = options?.takeIf { it.hasKey("orientation") && !it.isNull("orientation") }?.getBoolean("orientation"),
        )
    }

    private fun decodeBase64(value: String): ByteArray = Base64.decode(value.substringAfter("base64,", value), Base64.DEFAULT)

    private fun toWritableArray(results: List<TextResult>): WritableArray = Arguments.createArray().apply {
        results.forEach { result ->
            pushMap(Arguments.createMap().apply {
                putString("text", result.text); putDouble("score", result.score.toDouble())
                putArray("box_points", Arguments.createArray().apply { result.boxPoints.forEach { point -> pushArray(Arguments.createArray().apply { pushDouble(point.x.toDouble()); pushDouble(point.y.toDouble()) }) }) })
                putMap("frame", Arguments.createMap().apply { putDouble("width", result.frame.width.toDouble()); putDouble("height", result.frame.height.toDouble()); putDouble("top", result.frame.top.toDouble()); putDouble("left", result.frame.left.toDouble()) })
            })
        }
    }

}
