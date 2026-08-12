package com.byrizki.rusto.reactnative

import android.net.Uri
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Dynamic
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.ReadableType
import com.facebook.react.bridge.WritableArray
import com.byrizki.rusto.RustO
import com.byrizki.rusto.RustOConfig
import com.byrizki.rusto.TextResult
import java.io.File
import java.io.FileNotFoundException
import java.net.URLDecoder

class RustoModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private var rustoInstance: RustO? = null

    override fun getName(): String {
        return "Rusto"
    }

    @ReactMethod
    fun initialize(configOrDet: Dynamic, recModel: String?, dict: String?, promise: Promise) {
        try {
            rustoInstance?.close()
            rustoInstance = null

            if (configOrDet.type == ReadableType.Map) {
                val map = configOrDet.asMap()
                val config = parseConfigFromMap(map)
                rustoInstance = RustO.create(reactContext, config)
                promise.resolve(true)
                return
            }

            val detModelPath = if (configOrDet.type == ReadableType.String) configOrDet.asString() else "det.mnn"
            val recModelPath = recModel ?: "rec.mnn"
            val dictPath = dict ?: "dict.txt"

            rustoInstance = RustO.create(
                reactContext,
                detModelPath,
                recModelPath,
                dictPath
            )
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", "Failed to initialize RustO: ${e.message}", e)
        }
    }

    private fun parseConfigFromMap(map: ReadableMap): RustOConfig {
        return RustOConfig(
            detModelPath = if (map.hasKey("detModelPath")) map.getString("detModelPath") ?: "det.mnn" else "det.mnn",
            recModelPath = if (map.hasKey("recModelPath")) map.getString("recModelPath") ?: "rec.mnn" else "rec.mnn",
            dictPath = if (map.hasKey("dictPath")) map.getString("dictPath") ?: "dict.txt" else "dict.txt",
            clsModelPath = if (map.hasKey("clsModelPath")) map.getString("clsModelPath") else null,
            orientModelPath = if (map.hasKey("orientModelPath")) map.getString("orientModelPath") else null,
            unwarpModelPath = if (map.hasKey("unwarpModelPath")) map.getString("unwarpModelPath") else null,
            orientThreshold = if (map.hasKey("orientThreshold")) map.getDouble("orientThreshold").toFloat() else null,
            clsThreshold = if (map.hasKey("clsThreshold")) map.getDouble("clsThreshold").toFloat() else null,
            textScore = if (map.hasKey("textScore")) map.getDouble("textScore").toFloat() else 0.5f,
            detThresh = if (map.hasKey("detThresh")) map.getDouble("detThresh").toFloat() else 0.3f,
            detBoxThresh = if (map.hasKey("detBoxThresh")) map.getDouble("detBoxThresh").toFloat() else 0.5f,
            limitSideLen = if (map.hasKey("limitSideLen")) map.getInt("limitSideLen") else 736,
            limitType = if (map.hasKey("limitType")) map.getString("limitType") ?: "min" else "min",
            unclipRatio = if (map.hasKey("unclipRatio")) map.getDouble("unclipRatio").toFloat() else 2.0f,
            useDilation = if (map.hasKey("useDilation")) map.getBoolean("useDilation") else true,
            useDet = if (map.hasKey("useDet")) map.getBoolean("useDet") else true,
            useRec = if (map.hasKey("useRec")) map.getBoolean("useRec") else true,
            useCls = if (map.hasKey("useCls")) map.getBoolean("useCls") else false,
            useOrient = if (map.hasKey("useOrient")) map.getBoolean("useOrient") else false,
            useUnwarp = if (map.hasKey("useUnwarp")) map.getBoolean("useUnwarp") else false,
            debugImages = if (map.hasKey("debugImages")) map.getBoolean("debugImages") else false,
            minHeight = if (map.hasKey("minHeight")) map.getDouble("minHeight").toFloat() else 30.0f,
            maxSideLen = if (map.hasKey("maxSideLen")) map.getDouble("maxSideLen").toFloat() else 2000.0f,
            minSideLen = if (map.hasKey("minSideLen")) map.getDouble("minSideLen").toFloat() else 30.0f,
            returnWordBox = if (map.hasKey("returnWordBox")) map.getBoolean("returnWordBox") else false,
            returnSingleCharBox = if (map.hasKey("returnSingleCharBox")) map.getBoolean("returnSingleCharBox") else false,
            yThresholdMultiplier = if (map.hasKey("yThresholdMultiplier")) map.getDouble("yThresholdMultiplier").toFloat() else null,
            xThresholdMultiplier = if (map.hasKey("xThresholdMultiplier")) map.getDouble("xThresholdMultiplier").toFloat() else null
        )
    }

    private fun resolveImageBytesOrPath(imagePath: String, onPath: (String) -> Unit, onBytes: (ByteArray) -> Unit) {
        val trimmed = imagePath.trim()

        if (trimmed.startsWith("content://")) {
            val uri = Uri.parse(trimmed)
            val bytes = reactContext.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                ?: throw FileNotFoundException("Could not open content URI: $trimmed")
            onBytes(bytes)
            return
        }

        var cleanPath = trimmed
        if (cleanPath.startsWith("file://")) {
            val uri = Uri.parse(cleanPath)
            cleanPath = uri.path ?: cleanPath.removePrefix("file://")
        } else if (cleanPath.startsWith("file:")) {
            cleanPath = cleanPath.removePrefix("file:")
        }

        try {
            cleanPath = URLDecoder.decode(cleanPath, "UTF-8")
        } catch (_: Exception) {}

        val imageFile = File(cleanPath)
        if (imageFile.exists()) {
            onPath(cleanPath)
            return
        }

        try {
            val uri = Uri.parse(trimmed)
            val bytes = reactContext.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            if (bytes != null) {
                onBytes(bytes)
                return
            }
        } catch (_: Exception) {}

        throw FileNotFoundException("Image file not found: $imagePath (resolved path: $cleanPath)")
    }

    private fun performOcrForPath(imagePath: String, instance: RustO): List<TextResult> {
        var resultList: List<TextResult>? = null
        resolveImageBytesOrPath(
            imagePath,
            onPath = { resultList = instance.recognizeFile(it) },
            onBytes = { resultList = instance.recognize(it) }
        )
        return resultList ?: emptyList()
    }

    @ReactMethod
    fun detectText(imagePath: String, promise: Promise) {
        try {
            val instance = rustoInstance ?: run {
                promise.reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.")
                return
            }

            val results = performOcrForPath(imagePath, instance)
            val resultArray = convertResultsToWritableArray(results)
            promise.resolve(resultArray)
        } catch (e: FileNotFoundException) {
            promise.reject("FILE_NOT_FOUND", e.message ?: "Image file not found: $imagePath", e)
        } catch (e: Exception) {
            promise.reject("OCR_ERROR", "Failed to detect text: ${e.message}", e)
        }
    }

    @ReactMethod
    fun detectTextFromBytes(imageData: String, promise: Promise) {
        try {
            val instance = rustoInstance ?: run {
                promise.reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.")
                return
            }

            val imageBytes = android.util.Base64.decode(imageData, android.util.Base64.DEFAULT)
            val results = instance.recognize(imageBytes)
            val resultArray = convertResultsToWritableArray(results)
            promise.resolve(resultArray)
        } catch (e: Exception) {
            promise.reject("OCR_ERROR", "Failed to detect text from bytes: ${e.message}", e)
        }
    }

    @ReactMethod
    fun detectTextToRaw(imagePath: String, promise: Promise) {
        runOutputFormatForPath(imagePath, promise) { instance, path, bytes ->
            if (path != null) instance.recognizeFileToRaw(path) else instance.recognizeToRaw(bytes!!)
        }
    }

    @ReactMethod
    fun detectTextToCsv(imagePath: String, promise: Promise) {
        runOutputFormatForPath(imagePath, promise) { instance, path, bytes ->
            if (path != null) instance.recognizeFileToCsv(path) else instance.recognizeToCsv(bytes!!)
        }
    }

    @ReactMethod
    fun detectTextToTextWithPosition(imagePath: String, promise: Promise) {
        runOutputFormatForPath(imagePath, promise) { instance, path, bytes ->
            if (path != null) instance.recognizeFileToTextWithPosition(path) else instance.recognizeToTextWithPosition(bytes!!)
        }
    }

    @ReactMethod
    fun detectTextToSpatialText(
        imagePath: String,
        yThresholdMultiplier: Double,
        xThresholdMultiplier: Double,
        promise: Promise
    ) {
        runOutputFormatForPath(imagePath, promise) { instance, path, bytes ->
            val yMult = yThresholdMultiplier.toFloat()
            val xMult = xThresholdMultiplier.toFloat()
            if (path != null) {
                instance.recognizeFileToSpatialText(path, yMult, xMult)
            } else {
                instance.recognizeToSpatialText(bytes!!, yMult, xMult)
            }
        }
    }

    @ReactMethod
    fun detectTextFromBytesToRaw(imageData: String, promise: Promise) {
        runOutputFormatForBytes(imageData, promise) { instance, bytes ->
            instance.recognizeToRaw(bytes)
        }
    }

    @ReactMethod
    fun detectTextFromBytesToCsv(imageData: String, promise: Promise) {
        runOutputFormatForBytes(imageData, promise) { instance, bytes ->
            instance.recognizeToCsv(bytes)
        }
    }

    @ReactMethod
    fun detectTextFromBytesToTextWithPosition(imageData: String, promise: Promise) {
        runOutputFormatForBytes(imageData, promise) { instance, bytes ->
            instance.recognizeToTextWithPosition(bytes)
        }
    }

    @ReactMethod
    fun detectTextFromBytesToSpatialText(
        imageData: String,
        yThresholdMultiplier: Double,
        xThresholdMultiplier: Double,
        promise: Promise
    ) {
        runOutputFormatForBytes(imageData, promise) { instance, bytes ->
            instance.recognizeToSpatialText(bytes, yThresholdMultiplier.toFloat(), xThresholdMultiplier.toFloat())
        }
    }

    private inline fun runOutputFormatForPath(
        imagePath: String,
        promise: Promise,
        crossinline action: (RustO, String?, ByteArray?) -> String
    ) {
        try {
            val instance = rustoInstance ?: run {
                promise.reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.")
                return
            }

            var resultText = ""
            resolveImageBytesOrPath(
                imagePath,
                onPath = { resultText = action(instance, it, null) },
                onBytes = { resultText = action(instance, null, it) }
            )
            promise.resolve(resultText)
        } catch (e: FileNotFoundException) {
            promise.reject("FILE_NOT_FOUND", e.message ?: "Image file not found: $imagePath", e)
        } catch (e: Exception) {
            promise.reject("OCR_ERROR", "Failed to process text format: ${e.message}", e)
        }
    }

    private inline fun runOutputFormatForBytes(
        imageData: String,
        promise: Promise,
        action: (RustO, ByteArray) -> String
    ) {
        try {
            val instance = rustoInstance ?: run {
                promise.reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.")
                return
            }

            val imageBytes = android.util.Base64.decode(imageData, android.util.Base64.DEFAULT)
            val result = action(instance, imageBytes)
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("OCR_ERROR", "Failed to process text format from bytes: ${e.message}", e)
        }
    }

    @ReactMethod
    fun getVersion(promise: Promise) {
        try {
            val version = RustO.nativeVersion()
            promise.resolve(version)
        } catch (e: Exception) {
            promise.reject("VERSION_ERROR", "Failed to get version: ${e.message}", e)
        }
    }

    private fun convertResultsToWritableArray(results: List<TextResult>): WritableArray {
        val array = Arguments.createArray()
        
        for (result in results) {
            val map = Arguments.createMap()
            map.putString("text", result.text)
            map.putDouble("score", result.score.toDouble())
            
            val boxPoints = Arguments.createArray()
            for (point in result.boxPoints) {
                val pointArray = Arguments.createArray()
                pointArray.pushDouble(point.x.toDouble())
                pointArray.pushDouble(point.y.toDouble())
                boxPoints.pushArray(pointArray)
            }
            map.putArray("box_points", boxPoints)

            val frameMap = Arguments.createMap()
            frameMap.putDouble("width", result.frame.width.toDouble())
            frameMap.putDouble("height", result.frame.height.toDouble())
            frameMap.putDouble("top", result.frame.top.toDouble())
            frameMap.putDouble("left", result.frame.left.toDouble())
            map.putMap("frame", frameMap)
            
            array.pushMap(map)
        }
        
        return array
    }

    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        rustoInstance?.close()
        rustoInstance = null
    }
}
