package com.byrizki.rusto.reactnative

import android.util.Base64
import com.byrizki.rusto.ClassificationConfig
import com.byrizki.rusto.DetectTextResult
import com.byrizki.rusto.DetectionConfig
import com.byrizki.rusto.DetectionRunOptions
import com.byrizki.rusto.ImageSource
import com.byrizki.rusto.InitializeConfig
import com.byrizki.rusto.LayoutConfig
import com.byrizki.rusto.OcrRunOptions
import com.byrizki.rusto.OrientationConfig
import com.byrizki.rusto.OutputGranularity
import com.byrizki.rusto.PostDetectionOptions
import com.byrizki.rusto.PostRecognitionOptions
import com.byrizki.rusto.PreDetectionOptions
import com.byrizki.rusto.PreRecognitionOptions
import com.byrizki.rusto.RecognitionConfig
import com.byrizki.rusto.RecognitionRunOptions
import com.byrizki.rusto.RustO
import com.byrizki.rusto.TextResult
import com.byrizki.rusto.UnwarpConfig
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

    @ReactMethod
    fun close(promise: Promise) {
        try {
            rustoInstance?.close()
            rustoInstance = null
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("CLOSE_ERROR", "Failed to close RustO: ${e.message}", e)
        }
    }

    @ReactMethod
    fun isReady(promise: Promise) {
        promise.resolve(rustoInstance?.isReady() ?: false)
    }

    @ReactMethod
    fun version(promise: Promise) {
        promise.resolve(rustoInstance?.version() ?: RustO.version())
    }


    private fun parseInitializeConfig(map: ReadableMap?): InitializeConfig {
        if (map == null) return InitializeConfig()
        requireOnlyKeys(
            map,
            setOf("preset", "models", "recBatchNum", "model", "modelPath", "lang", "cacheDir", "enableReadingOrder", "layoutVariant"),
            "InitializeConfig contains an unknown key."
        )
        val template = if (map.hasKey("preset") && !map.isNull("preset")) {
            require(map.getType("preset") == ReadableType.String) { "InitializeConfig.preset must be a string." }
            map.getString("preset")!!.also {
                require(it in setOf("indra", "ppv6", "ppv5", "ppv4", "ppv3")) { "InitializeConfig.preset is invalid." }
            }
        } else null

        fun optString(key: String): String? {
            if (!map.hasKey(key) || map.isNull(key)) return null
            require(map.getType(key) == ReadableType.String) { "InitializeConfig.$key must be a string." }
            return map.getString(key)!!.also { require(it.isNotBlank()) { "InitializeConfig.$key must be non-empty." } }
        }

        fun optBoolean(key: String): Boolean? {
            if (!map.hasKey(key) || map.isNull(key)) return null
            require(map.getType(key) == ReadableType.Boolean) { "InitializeConfig.$key must be a boolean." }
            return map.getBoolean(key)
        }

        val recBatchNum = if (map.hasKey("recBatchNum") && !map.isNull("recBatchNum")) {
            require(map.getType("recBatchNum") == ReadableType.Number) { "InitializeConfig.recBatchNum must be a number." }
            map.getDouble("recBatchNum").also {
                require(it.isFinite() && it >= 1.0 && it % 1.0 == 0.0) { "InitializeConfig.recBatchNum must be an integer >= 1." }
            }.toInt()
        } else null

        val models = if (map.hasKey("models") && !map.isNull("models")) {
            require(map.getType("models") == ReadableType.Map) { "InitializeConfig.models must be an object." }
            map.getMap("models")!!
        } else null
        models?.let {
            requireOnlyKeys(
                it,
                setOf("detection", "recognition", "dictionary", "classification", "orientation", "unwarp", "layout"),
                "InitializeConfig.models contains an unknown key."
            )
        }
        fun model(name: String): String? {
            if (models == null || !models.hasKey(name) || models.isNull(name)) return null
            require(models.getType(name) == ReadableType.String) { "InitializeConfig.models.$name must be a string." }
            return models.getString(name)!!.also { require(it.isNotBlank()) { "InitializeConfig.models.$name must be non-empty." } }
        }
        val detModel = model("detection")
        val recModel = model("recognition")
        val dictPath = model("dictionary")

        return InitializeConfig(
            model = optString("model"),
            modelPath = optString("modelPath"),
            lang = optString("lang"),
            cacheDir = optString("cacheDir"),
            enableReadingOrder = optBoolean("enableReadingOrder"),
            layoutVariant = optString("layoutVariant"),
            template = template,
            detection = if (detModel != null) DetectionConfig(modelPath = detModel) else null,
            recognition = if (recModel != null || dictPath != null || recBatchNum != null) {
                RecognitionConfig(modelPath = recModel, dictPath = dictPath, recBatchNum = recBatchNum)
            } else null,
            classification = model("classification")?.let { ClassificationConfig(enabled = true, modelPath = it) },
            orientation = model("orientation")?.let { OrientationConfig(enabled = true, modelPath = it) },
            unwarp = model("unwarp")?.let { UnwarpConfig(enabled = true, modelPath = it) },
            layout = model("layout")?.let { LayoutConfig(enabled = true, modelPath = it) },
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
        requireOnlyKeys(
            options,
            setOf(
                "output",
                "autoUnwarp",
                "layoutReading",
                "orientationDetection",
                "lineClassification",
                "preDetection",
                "detection",
                "postDetection",
                "preRecognition",
                "recognition",
                "postRecognition"
            ),
            "OcrRunOptions contains an unknown key."
        )
        val output = if (options.hasKey("output") && !options.isNull("output")) {
            require(options.getType("output") == ReadableType.String) { "OcrRunOptions.output must be a string." }
            options.getString("output")!!
        } else "lines"
        require(output in setOf("lines", "words", "spatial")) { "OcrRunOptions.output is invalid." }

        fun boolean(name: String): Boolean? {
            if (!options.hasKey(name) || options.isNull(name)) return null
            require(options.getType(name) == ReadableType.Boolean) { "OcrRunOptions.$name must be a boolean." }
            return options.getBoolean(name)
        }

        fun subMap(name: String): ReadableMap? {
            if (!options.hasKey(name) || options.isNull(name)) return null
            require(options.getType(name) == ReadableType.Map) { "OcrRunOptions.$name must be an object." }
            return options.getMap(name)
        }

        return OcrRunOptions(
            output = OutputGranularity.entries.first { it.wireValue == output },
            autoUnwarp = boolean("autoUnwarp"),
            layoutReading = boolean("layoutReading"),
            orientationDetection = boolean("orientationDetection"),
            lineClassification = boolean("lineClassification"),
            preDetection = subMap("preDetection")?.let(::parsePreDetection),
            detection = subMap("detection")?.let(::parseRuntimeDetection),
            postDetection = subMap("postDetection")?.let(::parsePostDetection),
            preRecognition = subMap("preRecognition")?.let(::parsePreRecognition),
            recognition = subMap("recognition")?.let(::parseRecognition),
            postRecognition = subMap("postRecognition")?.let(::parsePostRecognition),
        )
    }

    private fun parsePreDetection(map: ReadableMap): PreDetectionOptions {
        requireOnlyKeys(map, setOf("descreenStrength", "targetMaxSide", "cropPaddingX", "cropPaddingY"), "OcrRunOptions.preDetection contains an unknown key.")
        fun nonNegFloat(key: String): Float? {
            if (!map.hasKey(key) || map.isNull(key)) return null
            require(map.getType(key) == ReadableType.Number) { "OcrRunOptions.preDetection.$key must be a number." }
            return map.getDouble(key).also { require(it.isFinite() && it >= 0.0) { "OcrRunOptions.preDetection.$key must be >= 0." } }.toFloat()
        }
        val targetMaxSide = if (!map.hasKey("targetMaxSide") || map.isNull("targetMaxSide")) null else {
            require(map.getType("targetMaxSide") == ReadableType.Number) { "OcrRunOptions.preDetection.targetMaxSide must be a number." }
            map.getDouble("targetMaxSide").also { require(it.isFinite() && it >= 1.0 && it % 1.0 == 0.0) { "OcrRunOptions.preDetection.targetMaxSide must be >= 1." } }.toLong().toUInt()
        }
        return PreDetectionOptions(
            descreenStrength = nonNegFloat("descreenStrength"),
            targetMaxSide = targetMaxSide,
            cropPaddingX = nonNegFloat("cropPaddingX"),
            cropPaddingY = nonNegFloat("cropPaddingY"),
        )
    }

    private fun parseRuntimeDetection(map: ReadableMap): DetectionRunOptions {
        requireOnlyKeys(map, setOf("limitSideLen", "limitType", "mean", "std"), "OcrRunOptions.detection contains an unknown key.")
        val side = if (!map.hasKey("limitSideLen") || map.isNull("limitSideLen")) null else {
            require(map.getType("limitSideLen") == ReadableType.Number) { "limitSideLen must be a number." }
            map.getDouble("limitSideLen").also { require(it.isFinite() && it in 1.0..32767.0 && it % 1.0 == 0.0) { "limitSideLen must be an integer between 1 and 32767." } }.toInt()
        }
        val type = if (!map.hasKey("limitType") || map.isNull("limitType")) null else map.getString("limitType")!!.also { require(it == "min" || it == "max") { "limitType is invalid." } }
        fun vector(name: String, nonzero: Boolean): FloatArray? {
            if (!map.hasKey(name) || map.isNull(name)) return null
            require(map.getType(name) == ReadableType.Array) { "$name must be an array." }
            val a = map.getArray(name)!!
            require(a.size() == 3) { "$name must contain three values." }
            return FloatArray(3) { i ->
                require(a.getType(i) == ReadableType.Number) { "$name must contain numbers." }
                a.getDouble(i).also { require(it.isFinite() && (!nonzero || it != 0.0)) { "$name contains invalid value." } }.toFloat()
            }
        }
        return DetectionRunOptions(side, type, vector("mean", false), vector("std", true))
    }

    private fun parsePostDetection(p: ReadableMap): PostDetectionOptions {
        requireOnlyKeys(
            p,
            setOf("threshold", "boxThreshold", "unclipRatio", "useDilation", "scoreMode", "maxCandidates", "minHeight", "minSideLen", "maxSideLen", "widthHeightRatio"),
            "OcrRunOptions.postDetection contains an unknown key."
        )
        fun score(n: String): Float? = if (!p.hasKey(n) || p.isNull(n)) null else {
            require(p.getType(n) == ReadableType.Number) { "$n must be a number." }
            p.getDouble(n).also { require(it.isFinite() && it in 0.0..1.0) { "$n is invalid." } }.toFloat()
        }
        fun posFloat(n: String): Float? = if (!p.hasKey(n) || p.isNull(n)) null else {
            require(p.getType(n) == ReadableType.Number) { "$n must be a number." }
            p.getDouble(n).also { require(it.isFinite() && it > 0.0) { "$n must be > 0." } }.toFloat()
        }
        val max = if (!p.hasKey("maxCandidates") || p.isNull("maxCandidates")) null else {
            require(p.getType("maxCandidates") == ReadableType.Number) { "maxCandidates must be a number." }
            p.getDouble("maxCandidates").also { require(it.isFinite() && it >= 1.0 && it % 1.0 == 0.0) { "maxCandidates is invalid." } }.toInt()
        }
        val dilation = if (!p.hasKey("useDilation") || p.isNull("useDilation")) null else {
            require(p.getType("useDilation") == ReadableType.Boolean) { "useDilation must be boolean." }
            p.getBoolean("useDilation")
        }
        val mode = if (!p.hasKey("scoreMode") || p.isNull("scoreMode")) null else p.getString("scoreMode")!!.also { require(it == "fast" || it == "slow") { "scoreMode is invalid." } }
        val minSide = posFloat("minSideLen")
        val maxSide = posFloat("maxSideLen")
        require(minSide == null || maxSide == null || minSide <= maxSide) { "minSideLen must be <= maxSideLen." }
        val ratio = if (!p.hasKey("widthHeightRatio") || p.isNull("widthHeightRatio")) null else {
            require(p.getType("widthHeightRatio") == ReadableType.Number) { "widthHeightRatio must be a number." }
            p.getDouble("widthHeightRatio").also { require(it.isFinite() && (it > 0.0 || it == -1.0)) { "widthHeightRatio is invalid." } }.toFloat()
        }
        return PostDetectionOptions(
            threshold = score("threshold"),
            boxThreshold = score("boxThreshold"),
            unclipRatio = posFloat("unclipRatio"),
            useDilation = dilation,
            scoreMode = mode,
            maxCandidates = max,
            minHeight = posFloat("minHeight"),
            minSideLen = minSide,
            maxSideLen = maxSide,
            widthHeightRatio = ratio,
        )
    }

    private fun parsePreRecognition(map: ReadableMap): PreRecognitionOptions {
        requireOnlyKeys(map, setOf("lineYThreshold", "wordXThreshold"), "OcrRunOptions.preRecognition contains an unknown key.")
        fun nonNeg(n: String): Float? = if (!map.hasKey(n) || map.isNull(n)) null else {
            require(map.getType(n) == ReadableType.Number) { "$n must be a number." }
            map.getDouble(n).also { require(it.isFinite() && it >= 0.0) { "$n must be >= 0." } }.toFloat()
        }
        return PreRecognitionOptions(
            lineYThreshold = nonNeg("lineYThreshold"),
            wordXThreshold = nonNeg("wordXThreshold"),
        )
    }

    private fun parseRecognition(map: ReadableMap): RecognitionRunOptions {
        requireOnlyKeys(map, setOf("batchSize"), "OcrRunOptions.recognition contains an unknown key.")
        val batchSize = if (!map.hasKey("batchSize") || map.isNull("batchSize")) null else {
            require(map.getType("batchSize") == ReadableType.Number) { "batchSize must be a number." }
            map.getDouble("batchSize").also { require(it.isFinite() && it >= 1.0 && it % 1.0 == 0.0) { "batchSize must be >= 1." } }.toInt()
        }
        return RecognitionRunOptions(batchSize = batchSize)
    }

    private fun parsePostRecognition(map: ReadableMap): PostRecognitionOptions {
        requireOnlyKeys(map, setOf("scoreThreshold"), "OcrRunOptions.postRecognition contains an unknown key.")
        val score = if (!map.hasKey("scoreThreshold") || map.isNull("scoreThreshold")) null else {
            require(map.getType("scoreThreshold") == ReadableType.Number) { "scoreThreshold must be a number." }
            map.getDouble("scoreThreshold").also { require(it.isFinite() && it in 0.0..1.0) { "scoreThreshold must be in [0, 1]." } }.toFloat()
        }
        return PostRecognitionOptions(scoreThreshold = score)
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
                result.regionId?.let { putInt("region_id", it) }
                result.layoutType?.let { putString("layout_type", it) }
            })
        }
    }
}
