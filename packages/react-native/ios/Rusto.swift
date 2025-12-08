import Foundation
import React

@objc(Rusto)
class RustoModule: NSObject {
    
    private var rustoInstance: OpaquePointer?
    
    @objc
    static func requiresMainQueueSetup() -> Bool {
        return false
    }
    
    @objc
    func initialize(_ detModel: String?,
                   recModel: String?,
                   dict: String?,
                   resolver resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        
        // Use default bundled models if not specified
        let detModelName = detModel ?? "det.mnn"
        let recModelName = recModel ?? "rec.mnn"
        let dictName = dict ?? "dict.txt"
        
        // Get paths from bundle or documents directory
        let detPath = getResourcePath(detModelName)
        let recPath = getResourcePath(recModelName)
        let dictPath = getResourcePath(dictName)
        
        guard let detPath = detPath, let recPath = recPath, let dictPath = dictPath else {
            reject("INIT_ERROR", "Failed to find model files: \(detModelName), \(recModelName), \(dictName)", nil)
            return
        }
        
        // Free existing instance if any
        if let existing = rustoInstance {
            rocr_free(existing)
        }
        
        // Initialize RustO with C FFI
        rustoInstance = rocr_new(detPath, recPath, dictPath)
        
        if rustoInstance == nil {
            reject("INIT_ERROR", "Failed to initialize RustO", nil)
        } else {
            resolve(true)
        }
    }
    
    @objc
    func detectText(_ imagePath: String,
                   resolver resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        
        guard let instance = rustoInstance else {
            reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.", nil)
            return
        }
        
        // Check if file exists
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: imagePath) else {
            reject("FILE_NOT_FOUND", "Image file not found: \(imagePath)", nil)
            return
        }
        
        // Call native OCR function
        var resultsPtr: UnsafeMutableRawPointer?
        var count: Int32 = 0
        
        let status = rocr_ocr_file(instance, imagePath, &resultsPtr, &count)
        
        if status != 0 {
            reject("OCR_ERROR", "OCR recognition failed with status: \(status)", nil)
            return
        }
        
        guard let results = resultsPtr else {
            reject("OCR_ERROR", "No results returned", nil)
            return
        }
        
        // Convert results to array
        let resultArray = convertResultsToArray(results, count: Int(count))
        
        // Free native results
        rocr_free_results(results, count)
        
        resolve(resultArray)
    }
    
    @objc
    func detectTextFromBytes(_ imageData: String,
                            resolver resolve: @escaping RCTPromiseResolveBlock,
                            rejecter reject: @escaping RCTPromiseRejectBlock) {
        
        guard let instance = rustoInstance else {
            reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.", nil)
            return
        }
        
        // Decode base64 image data
        guard let data = Data(base64Encoded: imageData) else {
            reject("DECODE_ERROR", "Failed to decode base64 image data", nil)
            return
        }
        
        // Call native OCR function
        var resultsPtr: UnsafeMutableRawPointer?
        var count: Int32 = 0
        
        data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            let status = rocr_ocr_data(instance, bytes.baseAddress, Int32(data.count), &resultsPtr, &count)
            
            if status != 0 {
                reject("OCR_ERROR", "OCR recognition failed with status: \(status)", nil)
                return
            }
        }
        
        guard let results = resultsPtr else {
            reject("OCR_ERROR", "No results returned", nil)
            return
        }
        
        // Convert results to array
        let resultArray = convertResultsToArray(results, count: Int(count))
        
        // Free native results
        rocr_free_results(results, count)
        
        resolve(resultArray)
    }
    
    @objc
    func getVersion(_ resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        
        if let version = rocr_version() {
            let versionString = String(cString: version)
            resolve(versionString)
        } else {
            reject("VERSION_ERROR", "Failed to get version", nil)
        }
    }
    
    // MARK: - Helper Methods
    
    private func getResourcePath(_ filename: String) -> String? {
        // First, check if it's an absolute path that exists
        if filename.hasPrefix("/") && FileManager.default.fileExists(atPath: filename) {
            return filename
        }
        
        // Try to find the file in bundle or documents, then copy to cache
        var sourcePath: String?
        
        // Try resource bundle first (for bundled models)
        if let bundlePath = Bundle(for: type(of: self)).path(forResource: "RustoModels", ofType: "bundle"),
           let bundle = Bundle(path: bundlePath) {
            let name = filename.replacingOccurrences(of: ".mnn", with: "").replacingOccurrences(of: ".txt", with: "")
            let ext = String(filename.split(separator: ".").last ?? "")
            sourcePath = bundle.path(forResource: name, ofType: ext)
        }
        
        // Try main bundle resources
        if sourcePath == nil {
            sourcePath = Bundle.main.path(forResource: filename, ofType: nil)
        }
        
        // Try documents directory
        if sourcePath == nil {
            let documentsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
            let filePath = (documentsPath as NSString).appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: filePath) {
                sourcePath = filePath
            }
        }
        
        // If we found the file, copy it to cache directory
        if let source = sourcePath {
            return copyToCache(source, filename: filename)
        }
        
        return nil
    }
    
    private func copyToCache(_ sourcePath: String, filename: String) -> String? {
        let cacheDir = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true)[0]
        let cachePath = (cacheDir as NSString).appendingPathComponent(filename)
        
        let fileManager = FileManager.default
        
        // If file already exists in cache, return cache path
        if fileManager.fileExists(atPath: cachePath) {
            return cachePath
        }
        
        // Copy file to cache
        do {
            try fileManager.copyItem(atPath: sourcePath, toPath: cachePath)
            return cachePath
        } catch {
            print("Error copying model to cache: \(error)")
            // If copy fails, return original source path
            return sourcePath
        }
    }
    
    private func convertResultsToArray(_ resultsPtr: UnsafeMutableRawPointer, count: Int) -> [[String: Any]] {
        var resultArray: [[String: Any]] = []
        let structSize = MemoryLayout<CTextResult>.stride
        
        for i in 0..<count {
            let offset = i * structSize
            let resultPtr = resultsPtr.advanced(by: offset)
            let result = resultPtr.load(as: CTextResult.self)
            
            if let textPtr = result.text {
                let textString = String(cString: textPtr)
                
                let resultDict: [String: Any] = [
                    "text": textString,
                    "score": Double(result.score),
                    "box_points": [
                        [Double(result.box_x1), Double(result.box_y1)],
                        [Double(result.box_x2), Double(result.box_y2)],
                        [Double(result.box_x3), Double(result.box_y3)],
                        [Double(result.box_x4), Double(result.box_y4)]
                    ]
                ]
                
                resultArray.append(resultDict)
            }
        }
        
        return resultArray
    }
    
    deinit {
        if let instance = rustoInstance {
            rocr_free(instance)
        }
    }
}

// MARK: - C FFI Declarations

// C struct matching Rust CTextResult
struct CTextResult {
    var text: UnsafeMutablePointer<CChar>?
    var score: Float
    var box_x1: Float
    var box_y1: Float
    var box_x2: Float
    var box_y2: Float
    var box_x3: Float
    var box_y3: Float
    var box_x4: Float
    var box_y4: Float
}

@_silgen_name("rocr_new")
func rocr_new(_ detModel: String, _ recModel: String, _ dict: String) -> OpaquePointer?

@_silgen_name("rocr_ocr_file")
func rocr_ocr_file(_ instance: OpaquePointer, _ imagePath: String, _ results: UnsafeMutablePointer<UnsafeMutableRawPointer?>, _ count: UnsafeMutablePointer<Int32>) -> Int32

@_silgen_name("rocr_ocr_data")
func rocr_ocr_data(_ instance: OpaquePointer, _ data: UnsafeRawPointer?, _ length: Int32, _ results: UnsafeMutablePointer<UnsafeMutableRawPointer?>, _ count: UnsafeMutablePointer<Int32>) -> Int32

@_silgen_name("rocr_free_results")
func rocr_free_results(_ results: UnsafeMutableRawPointer, _ count: Int32)

@_silgen_name("rocr_free")
func rocr_free(_ instance: OpaquePointer)

@_silgen_name("rocr_version")
func rocr_version() -> UnsafeMutablePointer<CChar>?
