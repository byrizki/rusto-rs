package com.byrizki.rusto.reactnative

import android.net.Uri
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import com.byrizki.rusto.RustO
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
    fun initialize(detModel: String?, recModel: String?, dict: String?, promise: Promise) {
        try {
            rustoInstance?.close()
            
            // Use default bundled models if not specified
            val detModelPath = detModel ?: "det.mnn"
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

    private fun performOcrForPath(imagePath: String, instance: RustO): List<TextResult> {
        val trimmed = imagePath.trim()
        
        // Handle content:// URI (e.g. from ImagePicker or media gallery)
        if (trimmed.startsWith("content://")) {
            val uri = Uri.parse(trimmed)
            val bytes = reactContext.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                ?: throw FileNotFoundException("Could not open content URI: $trimmed")
            return instance.recognize(bytes)
        }
        
        // Handle file:// or raw paths
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
            return instance.recognizeFile(cleanPath)
        }
        
        // Fallback: attempt to open as URI via ContentResolver
        try {
            val uri = Uri.parse(trimmed)
            val bytes = reactContext.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            if (bytes != null) {
                return instance.recognize(bytes)
            }
        } catch (_: Exception) {}

        throw FileNotFoundException("Image file not found: $imagePath (resolved path: $cleanPath)")
    }

    @ReactMethod
    fun detectText(imagePath: String, promise: Promise) {
        try {
            val instance = rustoInstance
            if (instance == null) {
                promise.reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.")
                return
            }

            val results = performOcrForPath(imagePath, instance)
            
            // Convert results to React Native format
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
            val instance = rustoInstance
            if (instance == null) {
                promise.reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.")
                return
            }

            // Decode base64 image data
            val imageBytes = android.util.Base64.decode(imageData, android.util.Base64.DEFAULT)
            
            // Perform OCR
            val results = instance.recognize(imageBytes)
            
            // Convert results to React Native format
            val resultArray = convertResultsToWritableArray(results)
            promise.resolve(resultArray)
        } catch (e: Exception) {
            promise.reject("OCR_ERROR", "Failed to detect text from bytes: ${e.message}", e)
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
