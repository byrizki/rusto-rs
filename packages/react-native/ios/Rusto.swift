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
    func initialize(_ configDict: [String: Any]?,
                   resolver resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        
        if let existing = rustoInstance {
            rocr_free(existing)
            rustoInstance = nil
        }
        
        let config = configDict ?? [:]
        let models = config["models"] as? [String: String] ?? [:]
        func model(_ key: String, fallback: String? = nil) -> String? {
            guard let value = models[key] ?? fallback else { return nil }
            return getResourcePath(value) ?? value
        }
        var resolvedConfig: [String: Any] = [
            "template": config["preset"] ?? "ppv6",
            "detection": ["modelPath": model("detection", fallback: "det.mnn")!],
            "recognition": [
                "modelPath": model("recognition", fallback: "rec.mnn")!,
                "dictPath": model("dictionary", fallback: "dict.txt")!,
            ],
        ]
        if let path = model("classification") { resolvedConfig["classification"] = ["enabled": true, "modelPath": path] }
        if let path = model("orientation") { resolvedConfig["orientation"] = ["enabled": true, "modelPath": path] }
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: resolvedConfig),
              let jsonStr = String(data: jsonData, encoding: .utf8) else {
            reject("INIT_ERROR", "Failed to serialize configuration", nil)
            return
        }
        
        rustoInstance = rocr_initialize(jsonStr)
        if rustoInstance == nil {
            reject("INIT_ERROR", "Failed to initialize RustO with config", nil)
        } else {
            resolve(nil)
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
    func detectText(_ source: [String: Any],
                    options: [String: Any]?,
                    resolver resolve: @escaping RCTPromiseResolveBlock,
                    rejecter reject: @escaping RCTPromiseRejectBlock) {
        guard let instance = rustoInstance else { reject("NOT_INITIALIZED", "RustO not initialized. Call initialize() first.", nil); return }
        let allowedKeys: Set<String> = ["uri", "base64"]
        guard source.count == 1,
              let key = source.keys.first,
              allowedKeys.contains(key),
              let value = source[key] as? String,
              !value.isEmpty else {
            reject("INVALID_SOURCE", "Provide exactly one non-empty source key: uri or base64.", nil)
            return
        }
        let entries = [(key, value)]
        let output = options?["output"] as? String ?? "lines"
        guard ["lines", "words", "spatial"].contains(output) else { reject("INVALID_OPTIONS", "Invalid output mode.", nil); return }
        var runtimeOptions = options ?? [:]
        runtimeOptions["output"] = output
        for key in ["lineYThreshold", "wordXThreshold"] {
            if let value = runtimeOptions[key] as? NSNumber, !value.doubleValue.isFinite || value.doubleValue < 0 { reject("INVALID_OPTIONS", "Invalid runtime options.", nil); return }
        }
        if let value = runtimeOptions["textScore"] as? NSNumber, !value.doubleValue.isFinite || value.doubleValue < 0 || value.doubleValue > 1 { reject("INVALID_OPTIONS", "Invalid runtime options.", nil); return }
        guard let jsonData = try? JSONSerialization.data(withJSONObject: runtimeOptions), let optionsJson = String(data: jsonData, encoding: .utf8) else { reject("INVALID_OPTIONS", "Invalid runtime options.", nil); return }

        if entries[0].0 == "base64" {
            let encoded = entries[0].1.components(separatedBy: "base64,").last ?? entries[0].1
            guard let data = Data(base64Encoded: encoded) else { reject("INVALID_SOURCE", "Invalid base64 image.", nil); return }
            if output == "spatial" {
                let text = data.withUnsafeBytes { rocr_detect_text_data_spatial(instance, $0.baseAddress, data.count, optionsJson) }
                guard let text else { reject("OCR_ERROR", "OCR recognition failed", nil); return }; defer { rocr_free_string(text) }; resolve(String(cString: text)); return
            }
            var resultsPtr: UnsafeMutableRawPointer?; var count: Int = 0
            let status = data.withUnsafeBytes { rocr_detect_text_data(instance, $0.baseAddress, data.count, optionsJson, &resultsPtr, &count) }
            guard status == 0, let results = resultsPtr else { reject("OCR_ERROR", "OCR recognition failed with status: \(status)", nil); return }
            defer { rocr_free_results(results, count) }; resolve(convertResultsToArray(results, count: count)); return
        }

        let rawPath = entries[0].1
        guard rawPath.hasPrefix("file:") || rawPath.hasPrefix("/") else { reject("INVALID_SOURCE", "Unsupported iOS URI scheme.", nil); return }
        let path = resolveFilePath(rawPath)
        guard FileManager.default.fileExists(atPath: path) else { reject("FILE_NOT_FOUND", "Image file not found: \(rawPath)", nil); return }
        if output == "spatial" {
            guard let text = rocr_detect_text_file_spatial(instance, path, optionsJson) else { reject("OCR_ERROR", "OCR recognition failed", nil); return }
            defer { rocr_free_string(text) }; resolve(String(cString: text)); return
        }
        var resultsPtr: UnsafeMutableRawPointer?; var count: Int = 0
        let status = rocr_detect_text_file(instance, path, optionsJson, &resultsPtr, &count)
        guard status == 0, let results = resultsPtr else { reject("OCR_ERROR", "OCR recognition failed with status: \(status)", nil); return }
        defer { rocr_free_results(results, count) }; resolve(convertResultsToArray(results, count: count))
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

@_silgen_name("rocr_initialize")
func rocr_initialize(_ configJson: String) -> OpaquePointer?

@_silgen_name("rocr_detect_text_file")
func rocr_detect_text_file(_ instance: OpaquePointer, _ imagePath: String, _ options: String, _ results: UnsafeMutablePointer<UnsafeMutableRawPointer?>, _ count: UnsafeMutablePointer<Int>) -> Int32

@_silgen_name("rocr_detect_text_data")
func rocr_detect_text_data(_ instance: OpaquePointer, _ data: UnsafeRawPointer?, _ length: Int, _ options: String, _ results: UnsafeMutablePointer<UnsafeMutableRawPointer?>, _ count: UnsafeMutablePointer<Int>) -> Int32

@_silgen_name("rocr_detect_text_file_spatial")
func rocr_detect_text_file_spatial(_ instance: OpaquePointer, _ imagePath: String, _ options: String) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_detect_text_data_spatial")
func rocr_detect_text_data_spatial(_ instance: OpaquePointer, _ data: UnsafeRawPointer?, _ length: Int, _ options: String) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_free_string")
func rocr_free_string(_ str: UnsafeMutablePointer<CChar>)

@_silgen_name("rocr_free_results")
func rocr_free_results(_ results: UnsafeMutableRawPointer, _ count: Int)

@_silgen_name("rocr_free")
func rocr_free(_ instance: OpaquePointer)
