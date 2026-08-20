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
        do {
            let config = try validatedInitializeConfig(configDict)
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
            guard let replacement = rocr_initialize(jsonStr) else {
                reject("INIT_ERROR", "Failed to initialize RustO with config", nil)
                return
            }
            if let existing = rustoInstance { rocr_free(existing) }
            rustoInstance = replacement
            resolve(nil)
        } catch {
            reject("INIT_ERROR", "Invalid initialization configuration: \(error.localizedDescription)", error)
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
              let rawValue = source[key] as? String else {
            reject("INVALID_SOURCE", "Provide exactly one non-empty source key: uri or base64.", nil)
            return
        }
        let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else {
            reject("INVALID_SOURCE", "Provide exactly one non-empty source key: uri or base64.", nil)
            return
        }
        let entries = [(key, value)]
        let runtimeOptions: [String: Any]
        do { runtimeOptions = try validatedRuntimeOptions(options) }
        catch { reject("INVALID_OPTIONS", "Invalid runtime options: \(error.localizedDescription)", error); return }
        let output = runtimeOptions["output"] as! String
        guard let jsonData = try? JSONSerialization.data(withJSONObject: runtimeOptions), let optionsJson = String(data: jsonData, encoding: .utf8) else { reject("INVALID_OPTIONS", "Invalid runtime options.", nil); return }

        if entries[0].0 == "base64" {
            let encoded = entries[0].1.components(separatedBy: "base64,").last ?? entries[0].1
            guard let data = Data(base64Encoded: encoded), !data.isEmpty else { reject("INVALID_SOURCE", "Base64 image must decode to non-empty data.", nil); return }
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

    private func validationError(_ message: String) -> NSError {
        NSError(domain: "react-native-rusto", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func validatedInitializeConfig(_ config: [String: Any]?) throws -> [String: Any] {
        let config = config ?? [:]
        guard !config.values.contains(where: { $0 is NSNull }) else { throw validationError("configuration values must not be null") }
        let allowed = Set(["preset", "models"])
        guard Set(config.keys).isSubset(of: allowed) else { throw validationError("unknown key") }
        if let preset = config["preset"] {
            guard let value = preset as? String, ["ppv6", "ppv5", "ppv4", "ppv3"].contains(value) else { throw validationError("preset is invalid") }
        }
        var validated = config
        if let rawModels = config["models"] {
            guard let models = rawModels as? [String: Any] else { throw validationError("models must be an object") }
            let allowedModels = Set(["detection", "recognition", "dictionary", "classification", "orientation"])
            guard Set(models.keys).isSubset(of: allowedModels) else { throw validationError("models contains an unknown key") }
            var strings: [String: String] = [:]
            for (key, value) in models {
                guard let path = value as? String, !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw validationError("models.\(key) must be a non-empty string") }
                strings[key] = path
            }
            validated["models"] = strings
        }
        return validated
    }

    private func validatedRuntimeOptions(_ input: [String: Any]?) throws -> [String: Any] {
        var options = input ?? [:]
        guard !options.values.contains(where: { $0 is NSNull }) else { throw validationError("options values must not be null") }
        let allowed = Set(["output", "lineYThreshold", "wordXThreshold", "textScore", "classification", "orientation", "preprocessing"])
        guard Set(options.keys).isSubset(of: allowed) else { throw validationError("unknown key") }
        if let output = options["output"] {
            guard let value = output as? String, ["lines", "words", "spatial"].contains(value) else { throw validationError("output is invalid") }
        } else { options["output"] = "lines" }
        for key in ["lineYThreshold", "wordXThreshold"] {
            if let value = options[key] {
                guard let number = value as? NSNumber, !(value is Bool), number.doubleValue.isFinite, number.doubleValue >= 0 else { throw validationError("\(key) is invalid") }
            }
        }
        if let value = options["textScore"] {
            guard let number = value as? NSNumber, !(value is Bool), number.doubleValue.isFinite, (0...1).contains(number.doubleValue) else { throw validationError("textScore is invalid") }
        }
        for key in ["classification", "orientation"] {
            if let value = options[key], !(value is Bool) { throw validationError("\(key) must be a boolean") }
        }
        if let rawPreprocessing = options["preprocessing"] {
            options["preprocessing"] = try validatedRuntimePreprocessing(rawPreprocessing)
        }
        return options
    }

    private func validatedRuntimePreprocessing(_ raw: Any) throws -> [String: Any] {
        guard let preprocessing = raw as? [String: Any] else { throw validationError("preprocessing must be an object") }
        guard Set(preprocessing.keys).isSubset(of: Set(["minHeight", "maxSideLen", "minSideLen", "widthHeightRatio", "detection"])) else { throw validationError("preprocessing contains an unknown key") }
        var result: [String: Any] = [:]
        func number(_ key: String, allowNegativeOne: Bool = false) throws -> NSNumber? {
            guard let raw = preprocessing[key] else { return nil }
            guard let value = raw as? NSNumber, !(raw is Bool), value.doubleValue.isFinite, value.doubleValue <= Double(Float.greatestFiniteMagnitude), value.doubleValue > 0 || (allowNegativeOne && value.doubleValue == -1) else { throw validationError("preprocessing.\(key) is invalid") }
            result[key] = value; return value
        }
        let minHeight = try number("minHeight"); let maxSideLen = try number("maxSideLen"); let minSideLen = try number("minSideLen"); _ = minHeight
        _ = try number("widthHeightRatio", allowNegativeOne: true)
        if let minSideLen, let maxSideLen, minSideLen.doubleValue > maxSideLen.doubleValue { throw validationError("preprocessing.minSideLen must be <= maxSideLen") }
        if let rawDetection = preprocessing["detection"] { result["detection"] = try validatedRuntimeDetection(rawDetection) }
        return result
    }

    private func validatedRuntimeDetection(_ raw: Any) throws -> [String: Any] {
        guard let detection = raw as? [String: Any] else { throw validationError("preprocessing.detection must be an object") }
        guard Set(detection.keys).isSubset(of: Set(["limitSideLen", "limitType", "mean", "std", "postprocess"])) else { throw validationError("preprocessing.detection contains an unknown key") }
        var result: [String: Any] = [:]
        if let raw = detection["limitSideLen"] { guard let value = raw as? NSNumber, !(raw is Bool), value.doubleValue.isFinite, value.doubleValue > 0, value.doubleValue <= 32767, value.doubleValue.rounded() == value.doubleValue else { throw validationError("limitSideLen is invalid") }; result["limitSideLen"] = value }
        if let raw = detection["limitType"] { guard let value = raw as? String, value == "min" || value == "max" else { throw validationError("limitType is invalid") }; result["limitType"] = value }
        for key in ["mean", "std"] {
            if let raw = detection[key] { guard let values = raw as? [Any], values.count == 3 else { throw validationError("\(key) must contain three numbers") }; for value in values { guard let number = value as? NSNumber, !(value is Bool), number.doubleValue.isFinite, key != "std" || number.doubleValue != 0 else { throw validationError("\(key) contains an invalid value") } }; result[key] = values }
        }
        if let rawPostprocess = detection["postprocess"] { result["postprocess"] = try validatedRuntimePostprocess(rawPostprocess) }
        return result
    }

    private func validatedRuntimePostprocess(_ raw: Any) throws -> [String: Any] {
        guard let postprocess = raw as? [String: Any] else { throw validationError("postprocess must be an object") }
        guard Set(postprocess.keys).isSubset(of: Set(["threshold", "boxThreshold", "maxCandidates", "unclipRatio", "useDilation"])) else { throw validationError("postprocess contains an unknown key") }
        var result: [String: Any] = [:]
        for key in ["threshold", "boxThreshold"] { if let raw = postprocess[key] { guard let value = raw as? NSNumber, !(raw is Bool), value.doubleValue.isFinite, (0...1).contains(value.doubleValue) else { throw validationError("\(key) is invalid") }; result[key] = value } }
        if let raw = postprocess["maxCandidates"] { guard let value = raw as? NSNumber, !(raw is Bool), value.doubleValue.isFinite, value.doubleValue >= 1, value.doubleValue.rounded() == value.doubleValue else { throw validationError("maxCandidates is invalid") }; result["maxCandidates"] = value }
        if let raw = postprocess["unclipRatio"] { guard let value = raw as? NSNumber, !(raw is Bool), value.doubleValue.isFinite, value.doubleValue > 0 else { throw validationError("unclipRatio is invalid") }; result["unclipRatio"] = value }
        if let raw = postprocess["useDilation"] { guard raw is Bool else { throw validationError("useDilation must be a boolean") }; result["useDilation"] = raw }
        return result
    }
    
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
