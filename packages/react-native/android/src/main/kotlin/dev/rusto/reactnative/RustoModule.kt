package dev.rusto.reactnative

import android.graphics.BitmapFactory
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import dev.rusto.RustO
import dev.rusto.TextResult
import java.io.File

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

    @ReactMethod
    fun detectText(imagePath: String, promise: Promise) {
        try {
            val instance = rustoInstance
            if (instance == null) {
                promise.reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.")
                return
            }

            // Check if file exists
            val imageFile = File(imagePath)
            if (!imageFile.exists()) {
                promise.reject("FILE_NOT_FOUND", "Image file not found: $imagePath")
                return
            }

            // Perform OCR
            val results = instance.recognizeFile(imagePath)
            
            // Convert results to React Native format
            val resultArray = convertResultsToWritableArray(results)
            promise.resolve(resultArray)
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
