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
    func initialize(_ configOrDet: Any?,
                   recModel: String?,
                   dict: String?,
                   resolver resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        
        if let configDict = configOrDet as? [String: Any] {
            if let existing = rustoInstance {
                rocr_free(existing)
                rustoInstance = nil
            }
            
            var resolvedConfig = configDict
            let detModelName = (configDict["detModelPath"] as? String) ?? "det.mnn"
            let recModelName = (configDict["recModelPath"] as? String) ?? "rec.mnn"
            let dictName = (configDict["dictPath"] as? String) ?? "dict.txt"
            
            resolvedConfig["detModelPath"] = getResourcePath(detModelName) ?? detModelName
            resolvedConfig["recModelPath"] = getResourcePath(recModelName) ?? recModelName
            resolvedConfig["dictPath"] = getResourcePath(dictName) ?? dictName
            
            if let cls = configDict["clsModelPath"] as? String {
                resolvedConfig["clsModelPath"] = getResourcePath(cls) ?? cls
            }
            if let orient = configDict["orientModelPath"] as? String {
                resolvedConfig["orientModelPath"] = getResourcePath(orient) ?? orient
            }
            if let unwarp = configDict["unwarpModelPath"] as? String {
                resolvedConfig["unwarpModelPath"] = getResourcePath(unwarp) ?? unwarp
            }
            
            guard let jsonData = try? JSONSerialization.data(withJSONObject: resolvedConfig),
                  let jsonStr = String(data: jsonData, encoding: .utf8) else {
                reject("INIT_ERROR", "Failed to serialize configuration", nil)
                return
            }
            
            rustoInstance = rocr_new_with_config(jsonStr)
            if rustoInstance == nil {
                reject("INIT_ERROR", "Failed to initialize RustO with config", nil)
            } else {
                resolve(true)
            }
            return
        }
        
        let detModelStr = configOrDet as? String
        let detModelName = detModelStr ?? "det.mnn"
        let recModelName = recModel ?? "rec.mnn"
        let dictName = dict ?? "dict.txt"
        
        let detPath = getResourcePath(detModelName)
        let recPath = getResourcePath(recModelName)
        let dictPath = getResourcePath(dictName)
        
        guard let detPath = detPath, let recPath = recPath, let dictPath = dictPath else {
            reject("INIT_ERROR", "Failed to find model files: \(detModelName), \(recModelName), \(dictName)", nil)
            return
        }
        
        if let existing = rustoInstance {
            rocr_free(existing)
            rustoInstance = nil
        }
        
        rustoInstance = rocr_new(detPath, recPath, dictPath)
        
        if rustoInstance == nil {
            reject("INIT_ERROR", "Failed to initialize RustO", nil)
        } else {
            resolve(true)
        }
    }
    
    private func resolveFilePath(_ rawPath: String) -> String {
        var path = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if path.hasPrefix("file://") {
            if let url = URL(string: path) {
                path = url.path
            } else {
                path = String(path.dropFirst(7))
            }
        } else if path.hasPrefix("file:") {
            path = String(path.dropFirst(5))
        }
        return path.removingPercentEncoding ?? path
    }
    
    @objc
    func detectText(_ imagePath: String,
                   resolver resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        
        guard let instance = rustoInstance else {
            reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.", nil)
            return
        }
        
        let resolvedPath = resolveFilePath(imagePath)
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: resolvedPath) else {
            reject("FILE_NOT_FOUND", "Image file not found: \(imagePath) (resolved path: \(resolvedPath))", nil)
            return
        }
        
        var resultsPtr: UnsafeMutableRawPointer?
        var count: Int32 = 0
        
        let status = rocr_ocr_file(instance, resolvedPath, &resultsPtr, &count)
        if status != 0 {
            reject("OCR_ERROR", "OCR recognition failed with status: \(status)", nil)
            return
        }
        
        guard let results = resultsPtr else {
            reject("OCR_ERROR", "No results returned", nil)
            return
        }
        
        let resultArray = convertResultsToArray(results, count: Int(count))
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
        
        guard let data = Data(base64Encoded: imageData) else {
            reject("DECODE_ERROR", "Failed to decode base64 image data", nil)
            return
        }
        
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
        
        let resultArray = convertResultsToArray(results, count: Int(count))
        rocr_free_results(results, count)
        resolve(resultArray)
    }
    
    @objc
    func detectTextToRaw(_ imagePath: String,
                         resolver resolve: @escaping RCTPromiseResolveBlock,
                         rejecter reject: @escaping RCTPromiseRejectBlock) {
        runOutputFormatForPath(imagePath, reject: reject) { output in
            guard let strPtr = rocr_output_to_raw(output) else { return "" }
            defer { rocr_free_string(strPtr) }
            return String(cString: strPtr)
        } resolve: { resolve($0) }
    }
    
    @objc
    func detectTextToCsv(_ imagePath: String,
                         resolver resolve: @escaping RCTPromiseResolveBlock,
                         rejecter reject: @escaping RCTPromiseRejectBlock) {
        runOutputFormatForPath(imagePath, reject: reject) { output in
            guard let strPtr = rocr_output_to_csv(output) else { return "" }
            defer { rocr_free_string(strPtr) }
            return String(cString: strPtr)
        } resolve: { resolve($0) }
    }
    
    @objc
    func detectTextToTextWithPosition(_ imagePath: String,
                                      resolver resolve: @escaping RCTPromiseResolveBlock,
                                      rejecter reject: @escaping RCTPromiseRejectBlock) {
        runOutputFormatForPath(imagePath, reject: reject) { output in
            guard let strPtr = rocr_output_to_text_with_position(output) else { return "" }
            defer { rocr_free_string(strPtr) }
            return String(cString: strPtr)
        } resolve: { resolve($0) }
    }
    
    @objc
    func detectTextToSpatialText(_ imagePath: String,
                                yThresholdMultiplier: NSNumber?,
                                xThresholdMultiplier: NSNumber?,
                                resolver resolve: @escaping RCTPromiseResolveBlock,
                                rejecter reject: @escaping RCTPromiseRejectBlock) {
        let yMult = yThresholdMultiplier?.floatValue ?? 0.0
        let xMult = xThresholdMultiplier?.floatValue ?? 0.0
        runOutputFormatForPath(imagePath, reject: reject) { output in
            guard let strPtr = rocr_output_to_spatial_text(output, yMult, xMult) else { return "" }
            defer { rocr_free_string(strPtr) }
            return String(cString: strPtr)
        } resolve: { resolve($0) }
    }
    
    @objc
    func detectTextFromBytesToRaw(_ imageData: String,
                                  resolver resolve: @escaping RCTPromiseResolveBlock,
                                  rejecter reject: @escaping RCTPromiseRejectBlock) {
        runOutputFormatForBytes(imageData, reject: reject) { output in
            guard let strPtr = rocr_output_to_raw(output) else { return "" }
            defer { rocr_free_string(strPtr) }
            return String(cString: strPtr)
        } resolve: { resolve($0) }
    }
    
    @objc
    func detectTextFromBytesToCsv(_ imageData: String,
                                  resolver resolve: @escaping RCTPromiseResolveBlock,
                                  rejecter reject: @escaping RCTPromiseRejectBlock) {
        runOutputFormatForBytes(imageData, reject: reject) { output in
            guard let strPtr = rocr_output_to_csv(output) else { return "" }
            defer { rocr_free_string(strPtr) }
            return String(cString: strPtr)
        } resolve: { resolve($0) }
    }
    
    @objc
    func detectTextFromBytesToTextWithPosition(_ imageData: String,
                                               resolver resolve: @escaping RCTPromiseResolveBlock,
                                               rejecter reject: @escaping RCTPromiseRejectBlock) {
        runOutputFormatForBytes(imageData, reject: reject) { output in
            guard let strPtr = rocr_output_to_text_with_position(output) else { return "" }
            defer { rocr_free_string(strPtr) }
            return String(cString: strPtr)
        } resolve: { resolve($0) }
    }
    
    @objc
    func detectTextFromBytesToSpatialText(_ imageData: String,
                                         yThresholdMultiplier: NSNumber?,
                                         xThresholdMultiplier: NSNumber?,
                                         resolver resolve: @escaping RCTPromiseResolveBlock,
                                         rejecter reject: @escaping RCTPromiseRejectBlock) {
        let yMult = yThresholdMultiplier?.floatValue ?? 0.0
        let xMult = xThresholdMultiplier?.floatValue ?? 0.0
        runOutputFormatForBytes(imageData, reject: reject) { output in
            guard let strPtr = rocr_output_to_spatial_text(output, yMult, xMult) else { return "" }
            defer { rocr_free_string(strPtr) }
            return String(cString: strPtr)
        } resolve: { resolve($0) }
    }
    
    private func runOutputFormatForPath(_ imagePath: String,
                                        reject: @escaping RCTPromiseRejectBlock,
                                        format: (OpaquePointer) -> String,
                                        resolve: @escaping (String) -> Void) {
        guard let instance = rustoInstance else {
            reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.", nil)
            return
        }
        let resolvedPath = resolveFilePath(imagePath)
        guard FileManager.default.fileExists(atPath: resolvedPath) else {
            reject("FILE_NOT_FOUND", "Image file not found: \(imagePath)", nil)
            return
        }
        
        var outputPtr: OpaquePointer?
        let status = rocr_ocr_file_with_output(instance, resolvedPath, &outputPtr)
        guard status == 0, let output = outputPtr else {
            reject("OCR_ERROR", "OCR failed with status: \(status)", nil)
            return
        }
        defer { rocr_free_output(output) }
        resolve(format(output))
    }
    
    private func runOutputFormatForBytes(_ imageData: String,
                                         reject: @escaping RCTPromiseRejectBlock,
                                         format: (OpaquePointer) -> String,
                                         resolve: @escaping (String) -> Void) {
        guard let instance = rustoInstance else {
            reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.", nil)
            return
        }
        guard let data = Data(base64Encoded: imageData) else {
            reject("DECODE_ERROR", "Failed to decode base64 image data", nil)
            return
        }
        
        var outputPtr: OpaquePointer?
        let status = data.withUnsafeBytes { bytes in
            rocr_ocr_data_with_output(instance, bytes.baseAddress, Int32(data.count), &outputPtr)
        }
        guard status == 0, let output = outputPtr else {
            reject("OCR_ERROR", "OCR failed with status: \(status)", nil)
            return
        }
        defer { rocr_free_output(output) }
        resolve(format(output))
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
        if filename.hasPrefix("/") && FileManager.default.fileExists(atPath: filename) {
            return filename
        }
        
        var sourcePath: String?
        let name = filename.replacingOccurrences(of: ".mnn", with: "").replacingOccurrences(of: ".txt", with: "")
        let ext = String(filename.split(separator: ".").last ?? "")
        
        let bundleNames = [
            "RustOModels", "RustoModels",
            "RustOModels_PPOCRv6_Tiny", "RustOModels_PPOCRv6_Small", "RustOModels_PPOCRv6_Medium",
            "RustOModels_PPOCRv5_Mobile", "RustOModels_PPOCRv5_Server",
            "RustOModels_PPOCRv4_Mobile", "RustOModels_PPOCRv4_Server"
        ]
        
        for bName in bundleNames {
            if let bundlePath = Bundle(for: type(of: self)).path(forResource: bName, ofType: "bundle") ?? Bundle.main.path(forResource: bName, ofType: "bundle"),
               let bundle = Bundle(path: bundlePath),
               let filePath = bundle.path(forResource: name, ofType: ext) {
                sourcePath = filePath
                break
            }
        }
        
        if sourcePath == nil {
            sourcePath = Bundle.main.path(forResource: filename, ofType: nil)
        }
        
        if sourcePath == nil {
            let documentsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
            let filePath = (documentsPath as NSString).appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: filePath) {
                sourcePath = filePath
            }
        }
        
        if let source = sourcePath {
            return copyToCache(source, filename: filename)
        }
        
        return nil
    }
    
    private func copyToCache(_ sourcePath: String, filename: String) -> String? {
        let cacheDir = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true)[0]
        let cachePath = (cacheDir as NSString).appendingPathComponent(filename)
        let fileManager = FileManager.default
        
        if fileManager.fileExists(atPath: cachePath) {
            return cachePath
        }
        
        do {
            try fileManager.copyItem(atPath: sourcePath, toPath: cachePath)
            return cachePath
        } catch {
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
                
                let minX = min(result.box_x1, result.box_x2, result.box_x3, result.box_x4)
                let maxX = max(result.box_x1, result.box_x2, result.box_x3, result.box_x4)
                let minY = min(result.box_y1, result.box_y2, result.box_y3, result.box_y4)
                let maxY = max(result.box_y1, result.box_y2, result.box_y3, result.box_y4)
                
                let resultDict: [String: Any] = [
                    "text": textString,
                    "score": Double(result.score),
                    "box_points": [
                        [Double(result.box_x1), Double(result.box_y1)],
                        [Double(result.box_x2), Double(result.box_y2)],
                        [Double(result.box_x3), Double(result.box_y3)],
                        [Double(result.box_x4), Double(result.box_y4)]
                    ],
                    "frame": [
                        "width": Double(maxX - minX),
                        "height": Double(maxY - minY),
                        "top": Double(minY),
                        "left": Double(minX)
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
    var frame_width: Float
    var frame_height: Float
    var frame_top: Float
    var frame_left: Float
}

@_silgen_name("rocr_new")
func rocr_new(_ detModel: String, _ recModel: String, _ dict: String) -> OpaquePointer?

@_silgen_name("rocr_new_with_config")
func rocr_new_with_config(_ configJson: String) -> OpaquePointer?

@_silgen_name("rocr_ocr_file")
func rocr_ocr_file(_ instance: OpaquePointer, _ imagePath: String, _ results: UnsafeMutablePointer<UnsafeMutableRawPointer?>, _ count: UnsafeMutablePointer<Int32>) -> Int32

@_silgen_name("rocr_ocr_data")
func rocr_ocr_data(_ instance: OpaquePointer, _ data: UnsafeRawPointer?, _ length: Int32, _ results: UnsafeMutablePointer<UnsafeMutableRawPointer?>, _ count: UnsafeMutablePointer<Int32>) -> Int32

@_silgen_name("rocr_ocr_file_with_output")
func rocr_ocr_file_with_output(_ instance: OpaquePointer, _ imagePath: String, _ output: UnsafeMutablePointer<OpaquePointer?>) -> Int32

@_silgen_name("rocr_ocr_data_with_output")
func rocr_ocr_data_with_output(_ instance: OpaquePointer, _ data: UnsafeRawPointer?, _ length: Int32, _ output: UnsafeMutablePointer<OpaquePointer?>) -> Int32

@_silgen_name("rocr_output_to_raw")
func rocr_output_to_raw(_ output: OpaquePointer) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_output_to_csv")
func rocr_output_to_csv(_ output: OpaquePointer) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_output_to_text_with_position")
func rocr_output_to_text_with_position(_ output: OpaquePointer) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_output_to_spatial_text")
func rocr_output_to_spatial_text(_ output: OpaquePointer, _ yThresholdMultiplier: Float, _ xThresholdMultiplier: Float) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_free_output")
func rocr_free_output(_ output: OpaquePointer)

@_silgen_name("rocr_free_string")
func rocr_free_string(_ str: UnsafeMutablePointer<CChar>)

@_silgen_name("rocr_free_results")
func rocr_free_results(_ results: UnsafeMutableRawPointer, _ count: Int32)

@_silgen_name("rocr_free")
func rocr_free(_ instance: OpaquePointer)

@_silgen_name("rocr_version")
func rocr_version() -> UnsafeMutablePointer<CChar>?
