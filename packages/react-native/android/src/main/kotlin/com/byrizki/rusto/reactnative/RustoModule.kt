package com.byrizki.rusto.reactnative

import android.util.Base64
import com.byrizki.rusto.ClassificationConfig
import com.byrizki.rusto.DetectionConfig
import com.byrizki.rusto.ImageSource
import com.byrizki.rusto.OcrRunOptions
import com.byrizki.rusto.OutputGranularity
import com.byrizki.rusto.DetectionRunOptions
import com.byrizki.rusto.PostprocessRunOptions
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
        requireOnlyKeys(options, setOf("output", "lineYThreshold", "wordXThreshold", "textScore", "classification", "orientation", "minHeight", "maxSideLen", "minSideLen", "widthHeightRatio", "detection", "postprocess"), "DetectTextOptions contains an unknown key.")
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
        fun resizeNumber(name: String): Float? { if (!options.hasKey(name) || options.isNull(name)) return null; require(options.getType(name) == ReadableType.Number) { "DetectTextOptions.$name must be a number." }; return options.getDouble(name).also { require(it.isFinite() && it > 0) { "DetectTextOptions.$name must be > 0." } }.toFloat() }
        val minHeight = resizeNumber("minHeight"); val maxSideLen = resizeNumber("maxSideLen"); val minSideLen = resizeNumber("minSideLen")
        require(minSideLen == null || maxSideLen == null || minSideLen <= maxSideLen) { "DetectTextOptions.minSideLen must be <= maxSideLen." }
        val ratio = if (!options.hasKey("widthHeightRatio") || options.isNull("widthHeightRatio")) null else { require(options.getType("widthHeightRatio") == ReadableType.Number) { "DetectTextOptions.widthHeightRatio must be a number." }; options.getDouble("widthHeightRatio").also { require(it.isFinite() && (it > 0 || it == -1.0)) { "DetectTextOptions.widthHeightRatio is invalid." } }.toFloat() }
        val detection = if (!options.hasKey("detection") || options.isNull("detection")) null else { require(options.getType("detection") == ReadableType.Map) { "DetectTextOptions.detection must be an object." }; parseRuntimeDetection(options.getMap("detection")!!) }
        val postprocess = if (!options.hasKey("postprocess") || options.isNull("postprocess")) null else { require(options.getType("postprocess") == ReadableType.Map) { "DetectTextOptions.postprocess must be an object." }; parseRuntimePostprocess(options.getMap("postprocess")!!) }
        return OcrRunOptions(OutputGranularity.entries.first { it.wireValue == output }, y?.toFloat(), x?.toFloat(), score?.toFloat(), boolean("classification"), boolean("orientation"), minHeight, maxSideLen, minSideLen, ratio, detection, postprocess)
    }

    private fun parseRuntimeDetection(map: ReadableMap): DetectionRunOptions {
        requireOnlyKeys(map, setOf("limitSideLen", "limitType", "mean", "std"), "DetectTextOptions.detection contains an unknown key.")
        val side = if (!map.hasKey("limitSideLen") || map.isNull("limitSideLen")) null else { require(map.getType("limitSideLen") == ReadableType.Number) { "limitSideLen must be a number." }; map.getDouble("limitSideLen").also { require(it.isFinite() && it in 1.0..32767.0 && it % 1 == 0.0) { "limitSideLen must be an integer between 1 and 32767." } }.toInt() }
        val type = if (!map.hasKey("limitType") || map.isNull("limitType")) null else map.getString("limitType")!!.also { require(it == "min" || it == "max") { "limitType is invalid." } }
        fun vector(name: String, nonzero: Boolean): FloatArray? { if (!map.hasKey(name) || map.isNull(name)) return null; require(map.getType(name) == ReadableType.Array) { "$name must be an array." }; val a = map.getArray(name)!!; require(a.size() == 3) { "$name must contain three values." }; return FloatArray(3) { i -> require(a.getType(i) == ReadableType.Number) { "$name must contain numbers." }; a.getDouble(i).also { require(it.isFinite() && (!nonzero || it != 0.0)) { "$name contains invalid value." } }.toFloat() } }
        return DetectionRunOptions(side, type, vector("mean", false), vector("std", true))
    }

    private fun parseRuntimePostprocess(p: ReadableMap): PostprocessRunOptions {
        requireOnlyKeys(p, setOf("threshold", "boxThreshold", "maxCandidates", "unclipRatio", "useDilation"), "DetectTextOptions.postprocess contains an unknown key.")
        fun score(n: String): Float? = if (!p.hasKey(n) || p.isNull(n)) null else p.getDouble(n).also { require(it.isFinite() && it in 0.0..1.0) { "$n is invalid." } }.toFloat()
        val max = if (!p.hasKey("maxCandidates") || p.isNull("maxCandidates")) null else p.getDouble("maxCandidates").also { require(it.isFinite() && it >= 1 && it % 1 == 0.0) { "maxCandidates is invalid." } }.toInt()
        val unclip = if (!p.hasKey("unclipRatio") || p.isNull("unclipRatio")) null else p.getDouble("unclipRatio").also { require(it.isFinite() && it > 0) { "unclipRatio is invalid." } }.toFloat()
        val dilation = if (!p.hasKey("useDilation") || p.isNull("useDilation")) null else { require(p.getType("useDilation") == ReadableType.Boolean) { "useDilation must be boolean." }; p.getBoolean("useDilation") }
        return PostprocessRunOptions(score("threshold"), score("boxThreshold"), max, unclip, dilation)
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
