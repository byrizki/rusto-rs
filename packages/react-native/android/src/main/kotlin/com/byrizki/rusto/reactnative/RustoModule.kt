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
import com.facebook.react.bridge.ReadableType
import com.facebook.react.bridge.WritableArray

class RustoModule(private val reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
    private var rustoInstance: RustO? = null
    override fun getName() = "Rusto"

    @ReactMethod
    fun initialize(configMap: ReadableMap?, promise: Promise) {
        try {
            val config = parseInitializeConfig(configMap)
            val replacement = RustO.initialize(reactContext, config)
            rustoInstance?.close()
            rustoInstance = replacement
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", "Failed to initialize RustO: ${e.message}", e)
        }
    }

    private fun parseInitializeConfig(map: ReadableMap?): InitializeConfig {
        if (map == null) return InitializeConfig()
        requireOnlyKeys(map, setOf("preset", "models"), "InitializeConfig contains an unknown key.")
        val template = if (map.hasKey("preset") && !map.isNull("preset")) {
            require(map.getType("preset") == ReadableType.String) { "InitializeConfig.preset must be a string." }
            map.getString("preset")!!.also { require(it in setOf("ppv6", "ppv5", "ppv4", "ppv3")) { "InitializeConfig.preset is invalid." } }
        } else "ppv6"
        val models = if (map.hasKey("models") && !map.isNull("models")) {
            require(map.getType("models") == ReadableType.Map) { "InitializeConfig.models must be an object." }
            map.getMap("models")!!
        } else null
        models?.let { requireOnlyKeys(it, setOf("detection", "recognition", "dictionary", "classification", "orientation"), "InitializeConfig.models contains an unknown key.") }
        fun model(name: String): String? {
            if (models == null || !models.hasKey(name) || models.isNull(name)) return null
            require(models.getType(name) == ReadableType.String) { "InitializeConfig.models.$name must be a string." }
            return models.getString(name)!!.also { require(it.isNotBlank()) { "InitializeConfig.models.$name must be non-empty." } }
        }
        return InitializeConfig(
            template = template,
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
            val code = when {
                e.message?.startsWith("Source ") == true -> "INVALID_SOURCE"
                e.message?.startsWith("Image file not found") == true -> "FILE_NOT_FOUND"
                else -> "INVALID_OPTIONS"
            }
            promise.reject(code, e.message, e)
        } catch (e: Exception) {
            promise.reject("OCR_ERROR", "Failed to detect text: ${e.message}", e)
        }
    }

    private fun requireOnlyKeys(map: ReadableMap, allowed: Set<String>, message: String) {
        val iterator = map.keySetIterator()
        while (iterator.hasNextKey()) require(iterator.nextKey() in allowed) { message }
    }

    private fun parseSource(source: ReadableMap): ImageSource {
        val allowedKeys = setOf("uri", "base64")
        val iterator = source.keySetIterator()
        val keys = mutableListOf<String>()
        while (iterator.hasNextKey()) keys += iterator.nextKey()
        require(keys.all { it in allowedKeys } && keys.size == 1) { "Source must contain exactly one key: uri or base64." }
        val sourceKey = keys.single()
        require(!source.isNull(sourceKey) && source.getType(sourceKey) == ReadableType.String) { "Source value must be a non-empty string." }
        val value = source.getString(sourceKey)!!.trim()
        require(value.isNotEmpty()) { "Source value must be a non-empty string." }
        return when (sourceKey) {
            "base64" -> ImageSource.Bytes(decodeBase64(value))
            else -> ImageSource.Uri(value)
        }
    }

    private fun parseOptions(options: ReadableMap?): OcrRunOptions {
        if (options == null) return OcrRunOptions()
        requireOnlyKeys(options, setOf("output", "lineYThreshold", "wordXThreshold", "textScore", "classification", "orientation"), "DetectTextOptions contains an unknown key.")
        val output = if (options.hasKey("output") && !options.isNull("output")) {
            require(options.getType("output") == ReadableType.String) { "DetectTextOptions.output must be a string." }
            options.getString("output")!!
        } else "lines"
        require(output in setOf("lines", "words", "spatial")) { "DetectTextOptions.output is invalid." }
        fun number(name: String): Double? {
            if (!options.hasKey(name) || options.isNull(name)) return null
            require(options.getType(name) == ReadableType.Number) { "DetectTextOptions.$name must be a number." }
            return options.getDouble(name)
        }
        fun boolean(name: String): Boolean? {
            if (!options.hasKey(name) || options.isNull(name)) return null
            require(options.getType(name) == ReadableType.Boolean) { "DetectTextOptions.$name must be a boolean." }
            return options.getBoolean(name)
        }
        val y = number("lineYThreshold")
        val x = number("wordXThreshold")
        val score = number("textScore")
        require((y == null || y.isFinite() && y >= 0) && (x == null || x.isFinite() && x >= 0) && (score == null || score.isFinite() && score in 0.0..1.0)) { "Invalid runtime options" }
        return OcrRunOptions(
            output = OutputGranularity.entries.first { it.wireValue == output },
            lineYThreshold = y?.toFloat(),
            wordXThreshold = x?.toFloat(),
            textScore = score?.toFloat(),
            classification = boolean("classification"),
            orientation = boolean("orientation"),
        )
    }

    private fun decodeBase64(value: String): ByteArray = try {
        Base64.decode(value.substringAfter("base64,", value), Base64.DEFAULT).also {
            require(it.isNotEmpty()) { "Source base64 must decode to non-empty data." }
        }
    } catch (e: IllegalArgumentException) {
        throw IllegalArgumentException("Source base64 is invalid.", e)
    }

    private fun toWritableArray(results: List<TextResult>): WritableArray = Arguments.createArray().apply {
        results.forEach { result ->
            pushMap(Arguments.createMap().apply {
                putString("text", result.text)
                putDouble("score", result.score.toDouble())
                putArray("box_points", Arguments.createArray().apply {
                    result.boxPoints.forEach { point ->
                        pushArray(Arguments.createArray().apply {
                            pushDouble(point.x.toDouble())
                            pushDouble(point.y.toDouble())
                        })
                    }
                })
                putMap("frame", Arguments.createMap().apply {
                    putDouble("width", result.frame.width.toDouble())
                    putDouble("height", result.frame.height.toDouble())
                    putDouble("top", result.frame.top.toDouble())
                    putDouble("left", result.frame.left.toDouble())
                })
            })
        }
    }

}
