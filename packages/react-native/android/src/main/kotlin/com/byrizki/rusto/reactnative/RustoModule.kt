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
import com.byrizki.rusto.ClassificationConfig
import com.byrizki.rusto.DetectionConfig
import com.byrizki.rusto.LayoutConfig
import com.byrizki.rusto.OrientationConfig
import com.byrizki.rusto.PreprocessingConfig
import com.byrizki.rusto.RecognitionConfig
import com.byrizki.rusto.RustO
import com.byrizki.rusto.RustOConfig
import com.byrizki.rusto.TextResult
import com.byrizki.rusto.UnwarpConfig
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
    fun initialize(configMap: ReadableMap?, promise: Promise) {
        try {
            rustoInstance?.close()
            rustoInstance = null

            val config = if (configMap != null) parseConfigFromMap(configMap) else RustOConfig()
            rustoInstance = RustO.create(reactContext, config)
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", "Failed to initialize RustO: ${e.message}", e)
        }
    }

    private fun parseConfigFromMap(map: ReadableMap): RustOConfig {
        val detMap = if (map.hasKey("detection") && !map.isNull("detection")) map.getMap("detection") else null
        val detection = detMap?.let {
            DetectionConfig(
                enabled = if (it.hasKey("enabled") && !it.isNull("enabled")) it.getBoolean("enabled") else null,
                modelPath = if (it.hasKey("modelPath") && !it.isNull("modelPath")) it.getString("modelPath") else null,
                thresh = if (it.hasKey("thresh") && !it.isNull("thresh")) it.getDouble("thresh").toFloat() else null,
                boxThresh = if (it.hasKey("boxThresh") && !it.isNull("boxThresh")) it.getDouble("boxThresh").toFloat() else null,
                unclipRatio = if (it.hasKey("unclipRatio") && !it.isNull("unclipRatio")) it.getDouble("unclipRatio").toFloat() else null,
                limitSideLen = if (it.hasKey("limitSideLen") && !it.isNull("limitSideLen")) it.getInt("limitSideLen") else null,
                limitType = if (it.hasKey("limitType") && !it.isNull("limitType")) it.getString("limitType") else null,
                useDilation = if (it.hasKey("useDilation") && !it.isNull("useDilation")) it.getBoolean("useDilation") else null
            )
        }

        val recMap = if (map.hasKey("recognition") && !map.isNull("recognition")) map.getMap("recognition") else null
        val recognition = recMap?.let {
            RecognitionConfig(
                enabled = if (it.hasKey("enabled") && !it.isNull("enabled")) it.getBoolean("enabled") else null,
                modelPath = if (it.hasKey("modelPath") && !it.isNull("modelPath")) it.getString("modelPath") else null,
                dictPath = if (it.hasKey("dictPath") && !it.isNull("dictPath")) it.getString("dictPath") else null,
                scoreThresh = if (it.hasKey("scoreThresh") && !it.isNull("scoreThresh")) it.getDouble("scoreThresh").toFloat() else null,
                returnWordBox = if (it.hasKey("returnWordBox") && !it.isNull("returnWordBox")) it.getBoolean("returnWordBox") else null,
                returnSingleCharBox = if (it.hasKey("returnSingleCharBox") && !it.isNull("returnSingleCharBox")) it.getBoolean("returnSingleCharBox") else null
            )
        }

        val clsMap = if (map.hasKey("classification") && !map.isNull("classification")) map.getMap("classification") else null
        val classification = clsMap?.let {
            ClassificationConfig(
                enabled = if (it.hasKey("enabled") && !it.isNull("enabled")) it.getBoolean("enabled") else null,
                modelPath = if (it.hasKey("modelPath") && !it.isNull("modelPath")) it.getString("modelPath") else null,
                thresh = if (it.hasKey("thresh") && !it.isNull("thresh")) it.getDouble("thresh").toFloat() else null
            )
        }

        val orientMap = if (map.hasKey("orientation") && !map.isNull("orientation")) map.getMap("orientation") else null
        val orientation = orientMap?.let {
            OrientationConfig(
                enabled = if (it.hasKey("enabled") && !it.isNull("enabled")) it.getBoolean("enabled") else null,
                modelPath = if (it.hasKey("modelPath") && !it.isNull("modelPath")) it.getString("modelPath") else null,
                thresh = if (it.hasKey("thresh") && !it.isNull("thresh")) it.getDouble("thresh").toFloat() else null
            )
        }

        val unwarpMap = if (map.hasKey("unwarp") && !map.isNull("unwarp")) map.getMap("unwarp") else null
        val unwarp = unwarpMap?.let {
            UnwarpConfig(
                enabled = if (it.hasKey("enabled") && !it.isNull("enabled")) it.getBoolean("enabled") else null,
                modelPath = if (it.hasKey("modelPath") && !it.isNull("modelPath")) it.getString("modelPath") else null
            )
        }

        val prepMap = if (map.hasKey("preprocessing") && !map.isNull("preprocessing")) map.getMap("preprocessing") else null
        val preprocessing = prepMap?.let {
            PreprocessingConfig(
                minHeight = if (it.hasKey("minHeight") && !it.isNull("minHeight")) it.getDouble("minHeight").toFloat() else null,
                maxSideLen = if (it.hasKey("maxSideLen") && !it.isNull("maxSideLen")) it.getDouble("maxSideLen").toFloat() else null,
                minSideLen = if (it.hasKey("minSideLen") && !it.isNull("minSideLen")) it.getDouble("minSideLen").toFloat() else null,
                debugImages = if (it.hasKey("debugImages") && !it.isNull("debugImages")) it.getBoolean("debugImages") else null
            )
        }

        val layoutMap = if (map.hasKey("layout") && !map.isNull("layout")) map.getMap("layout") else null
        val layout = layoutMap?.let {
            LayoutConfig(
                yThresholdMultiplier = if (it.hasKey("yThresholdMultiplier") && !it.isNull("yThresholdMultiplier")) it.getDouble("yThresholdMultiplier").toFloat() else null,
                xThresholdMultiplier = if (it.hasKey("xThresholdMultiplier") && !it.isNull("xThresholdMultiplier")) it.getDouble("xThresholdMultiplier").toFloat() else null
            )
        }

        return RustOConfig(
            template = if (map.hasKey("template") && !map.isNull("template")) map.getString("template") else null,
            detection = detection,
            recognition = recognition,
            classification = classification,
            orientation = orientation,
            unwarp = unwarp,
            preprocessing = preprocessing,
            layout = layout
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
