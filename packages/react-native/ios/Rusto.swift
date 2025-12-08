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
            rusto_free(existing)
        }
        
        // Initialize RustO with C FFI
        rustoInstance = rusto_new(detPath, recPath, dictPath)
        
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
        
        let status = rusto_ocr_file(instance, imagePath, &resultsPtr, &count)
        
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
        rusto_free_results(results, count)
        
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
            let status = rusto_ocr_data(instance, bytes.baseAddress, Int32(data.count), &resultsPtr, &count)
            
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
        rusto_free_results(results, count)
        
        resolve(resultArray)
    }
    
    @objc
    func getVersion(_ resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        
        if let version = rusto_version() {
            let versionString = String(cString: version)
            resolve(versionString)
        } else {
            reject("VERSION_ERROR", "Failed to get version", nil)
        }
    }
    
    // MARK: - Helper Methods
    
    private func getResourcePath(_ filename: String) -> String? {
        // Try resource bundle first (for bundled models)
        if let bundlePath = Bundle(for: type(of: self)).path(forResource: "RustoModels", ofType: "bundle"),
           let bundle = Bundle(path: bundlePath),
           let filePath = bundle.path(forResource: filename.replacingOccurrences(of: ".mnn", with: "").replacingOccurrences(of: ".txt", with: ""), ofType: String(filename.split(separator: ".").last ?? "")) {
            return filePath
        }
        
        // Try main bundle resources
        if let bundlePath = Bundle.main.path(forResource: filename, ofType: nil) {
            return bundlePath
        }
        
        // Try documents directory
        let documentsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
        let filePath = (documentsPath as NSString).appendingPathComponent(filename)
        
        if FileManager.default.fileExists(atPath: filePath) {
            return filePath
        }
        
        return nil
    }
    
    private func convertResultsToArray(_ resultsPtr: UnsafeMutableRawPointer, count: Int) -> [[String: Any]] {
        var resultArray: [[String: Any]] = []
        
        for i in 0..<count {
            var text: UnsafeMutablePointer<CChar>?
            var score: Float = 0.0
            var boxPoints = [Float](repeating: 0, count: 8)
            
            rusto_get_result(resultsPtr, Int32(i), &text, &score, &boxPoints)
            
            if let textPtr = text {
                let textString = String(cString: textPtr)
                
                let result: [String: Any] = [
                    "text": textString,
                    "score": Double(score),
                    "box_points": [
                        [Double(boxPoints[0]), Double(boxPoints[1])],
                        [Double(boxPoints[2]), Double(boxPoints[3])],
                        [Double(boxPoints[4]), Double(boxPoints[5])],
                        [Double(boxPoints[6]), Double(boxPoints[7])]
                    ]
                ]
                
                resultArray.append(result)
            }
        }
        
        return resultArray
    }
    
    deinit {
        if let instance = rustoInstance {
            rusto_free(instance)
        }
    }
}

// MARK: - C FFI Declarations

@_silgen_name("rusto_new")
func rusto_new(_ detModel: String, _ recModel: String, _ dict: String) -> OpaquePointer?

@_silgen_name("rusto_ocr_file")
func rusto_ocr_file(_ instance: OpaquePointer, _ imagePath: String, _ results: UnsafeMutablePointer<UnsafeMutableRawPointer?>, _ count: UnsafeMutablePointer<Int32>) -> Int32

@_silgen_name("rusto_ocr_data")
func rusto_ocr_data(_ instance: OpaquePointer, _ data: UnsafeRawPointer?, _ length: Int32, _ results: UnsafeMutablePointer<UnsafeMutableRawPointer?>, _ count: UnsafeMutablePointer<Int32>) -> Int32

@_silgen_name("rusto_get_result")
func rusto_get_result(_ results: UnsafeMutableRawPointer, _ index: Int32, _ text: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>, _ score: UnsafeMutablePointer<Float>, _ boxPoints: UnsafeMutablePointer<Float>)

@_silgen_name("rusto_free_results")
func rusto_free_results(_ results: UnsafeMutableRawPointer, _ count: Int32)

@_silgen_name("rusto_free")
func rusto_free(_ instance: OpaquePointer)

@_silgen_name("rusto_version")
func rusto_version() -> UnsafeMutablePointer<CChar>?
