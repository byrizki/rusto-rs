import Foundation

public struct Point2D {
    public let x: Float
    public let y: Float
    
    public init(x: Float, y: Float) {
        self.x = x
        self.y = y
    }
}

public struct Frame {
    public let width: Float
    public let height: Float
    public let top: Float
    public let left: Float
    
    public init(width: Float, height: Float, top: Float, left: Float) {
        self.width = width
        self.height = height
        self.top = top
        self.left = left
    }
}

public struct TextResult {
    public let text: String
    public let score: Float
    public let boxPoints: [Point2D]
    public let frame: Frame
    
    public init(text: String, score: Float, boxPoints: [Point2D], frame: Frame? = nil) {
        self.text = text
        self.score = score
        self.boxPoints = boxPoints
        if let frame = frame {
            self.frame = frame
        } else {
            let minX = boxPoints.map { $0.x }.min() ?? 0.0
            let maxX = boxPoints.map { $0.x }.max() ?? 0.0
            let minY = boxPoints.map { $0.y }.min() ?? 0.0
            let maxY = boxPoints.map { $0.y }.max() ?? 0.0
            self.frame = Frame(width: maxX - minX, height: maxY - minY, top: minY, left: minX)
        }
    }
}

public struct DetectionConfig: Codable {
    public var enabled: Bool? = nil
    public var modelPath: String? = nil
    public var thresh: Float? = nil
    public var boxThresh: Float? = nil
    public var unclipRatio: Float? = nil
    public var limitSideLen: Int? = nil
    public var limitType: String? = nil
    public var useDilation: Bool? = nil

    public init(enabled: Bool? = nil, modelPath: String? = nil, thresh: Float? = nil, boxThresh: Float? = nil, unclipRatio: Float? = nil, limitSideLen: Int? = nil, limitType: String? = nil, useDilation: Bool? = nil) {
        self.enabled = enabled
        self.modelPath = modelPath
        self.thresh = thresh
        self.boxThresh = boxThresh
        self.unclipRatio = unclipRatio
        self.limitSideLen = limitSideLen
        self.limitType = limitType
        self.useDilation = useDilation
    }
}

public struct RecognitionConfig: Codable {
    public var enabled: Bool? = nil
    public var modelPath: String? = nil
    public var dictPath: String? = nil
    public var scoreThresh: Float? = nil
    public var returnWordBox: Bool? = nil
    public var returnSingleCharBox: Bool? = nil

    public init(enabled: Bool? = nil, modelPath: String? = nil, dictPath: String? = nil, scoreThresh: Float? = nil, returnWordBox: Bool? = nil, returnSingleCharBox: Bool? = nil) {
        self.enabled = enabled
        self.modelPath = modelPath
        self.dictPath = dictPath
        self.scoreThresh = scoreThresh
        self.returnWordBox = returnWordBox
        self.returnSingleCharBox = returnSingleCharBox
    }
}

/// NOTE: Available ONLY on PP-OCRv4 and PP-OCRv5
public struct ClassificationConfig: Codable {
    public var enabled: Bool? = nil
    public var modelPath: String? = nil
    public var thresh: Float? = nil

    public init(enabled: Bool? = nil, modelPath: String? = nil, thresh: Float? = nil) {
        self.enabled = enabled
        self.modelPath = modelPath
        self.thresh = thresh
    }
}

public struct OrientationConfig: Codable {
    public var enabled: Bool? = nil
    public var modelPath: String? = nil
    public var thresh: Float? = nil

    public init(enabled: Bool? = nil, modelPath: String? = nil, thresh: Float? = nil) {
        self.enabled = enabled
        self.modelPath = modelPath
        self.thresh = thresh
    }
}

public struct UnwarpConfig: Codable {
    public var enabled: Bool? = nil
    public var modelPath: String? = nil

    public init(enabled: Bool? = nil, modelPath: String? = nil) {
        self.enabled = enabled
        self.modelPath = modelPath
    }
}

public struct LayoutConfig: Codable {
    public var yThresholdMultiplier: Float? = nil
    public var xThresholdMultiplier: Float? = nil

    public init(yThresholdMultiplier: Float? = nil, xThresholdMultiplier: Float? = nil) {
        self.yThresholdMultiplier = yThresholdMultiplier
        self.xThresholdMultiplier = xThresholdMultiplier
    }
}

public struct InitializeConfig: Codable {
    public var template: String? = nil
    public var detection: DetectionConfig? = nil
    public var recognition: RecognitionConfig? = nil
    public var classification: ClassificationConfig? = nil
    public var orientation: OrientationConfig? = nil
    public var unwarp: UnwarpConfig? = nil
    public var layout: LayoutConfig? = nil

    public init(
        template: String? = nil,
        detection: DetectionConfig? = nil,
        recognition: RecognitionConfig? = nil,
        classification: ClassificationConfig? = nil,
        orientation: OrientationConfig? = nil,
        unwarp: UnwarpConfig? = nil,
        layout: LayoutConfig? = nil
    ) {
        self.template = template
        self.detection = detection
        self.recognition = recognition
        self.classification = classification
        self.orientation = orientation
        self.unwarp = unwarp
        self.layout = layout
    }

    public static func ppv6(detection: DetectionConfig? = nil, recognition: RecognitionConfig? = nil) -> InitializeConfig {
        InitializeConfig(template: "ppv6", detection: detection, recognition: recognition)
    }

    public static func ppv5(detection: DetectionConfig? = nil, recognition: RecognitionConfig? = nil) -> InitializeConfig {
        InitializeConfig(template: "ppv5", detection: detection, recognition: recognition)
    }

    public static func ppv4(detection: DetectionConfig? = nil, recognition: RecognitionConfig? = nil) -> InitializeConfig {
        InitializeConfig(template: "ppv4", detection: detection, recognition: recognition)
    }

    public static func ppv3(detection: DetectionConfig? = nil, recognition: RecognitionConfig? = nil) -> InitializeConfig {
        InitializeConfig(template: "ppv3", detection: detection, recognition: recognition)
    }
}

public enum RustOError: Error {
    case initializationFailed
    case recognitionFailed(Int32)
    case invalidHandle
    case invalidPath
    case serializationFailed
}

public enum OutputGranularity: String, Codable {
    case lines, words, spatial
}

public struct PostprocessRunOptions: Codable {
    public var threshold: Float?
    public var boxThreshold: Float?
    public var maxCandidates: Int?
    public var unclipRatio: Float?
    public var useDilation: Bool?
    public init(threshold: Float? = nil, boxThreshold: Float? = nil, maxCandidates: Int? = nil, unclipRatio: Float? = nil, useDilation: Bool? = nil) { self.threshold = threshold; self.boxThreshold = boxThreshold; self.maxCandidates = maxCandidates; self.unclipRatio = unclipRatio; self.useDilation = useDilation }
}

public struct DetectionRunOptions: Codable {
    public var limitSideLen: Int?
    public var limitType: String?
    public var mean: [Float]?
    public var std: [Float]?
    public var postprocess: PostprocessRunOptions?
    public init(limitSideLen: Int? = nil, limitType: String? = nil, mean: [Float]? = nil, std: [Float]? = nil, postprocess: PostprocessRunOptions? = nil) { self.limitSideLen = limitSideLen; self.limitType = limitType; self.mean = mean; self.std = std; self.postprocess = postprocess }
}

public struct PreprocessingRunOptions: Codable {
    public var minHeight: Float?
    public var maxSideLen: Float?
    public var minSideLen: Float?
    public var widthHeightRatio: Float?
    public var detection: DetectionRunOptions?
    public init(minHeight: Float? = nil, maxSideLen: Float? = nil, minSideLen: Float? = nil, widthHeightRatio: Float? = nil, detection: DetectionRunOptions? = nil) { self.minHeight = minHeight; self.maxSideLen = maxSideLen; self.minSideLen = minSideLen; self.widthHeightRatio = widthHeightRatio; self.detection = detection }
}

public struct OcrRunOptions: Codable {
    public var output: OutputGranularity
    public var lineYThreshold: Float?
    public var wordXThreshold: Float?
    public var textScore: Float?
    public var classification: Bool?
    public var orientation: Bool?
    public var preprocessing: PreprocessingRunOptions?

    public init(output: OutputGranularity = .lines, lineYThreshold: Float? = nil, wordXThreshold: Float? = nil, textScore: Float? = nil, classification: Bool? = nil, orientation: Bool? = nil, preprocessing: PreprocessingRunOptions? = nil) {
        self.output = output; self.lineYThreshold = lineYThreshold; self.wordXThreshold = wordXThreshold; self.textScore = textScore; self.classification = classification; self.orientation = orientation; self.preprocessing = preprocessing
    }
}

public enum ImageSource {
    case uri(String)
    case bytes(Data)
}

public enum DetectTextResult {
    case structured([TextResult])
    case spatial(String)
}

public class RustO {
    private var handle: OpaquePointer?

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    public static var version: String {
        guard let versionPtr = rocr_version() else { return "unknown" }
        return String(cString: versionPtr)
    }

    public static func initialize(config: InitializeConfig = InitializeConfig()) throws -> RustO {
        var resolvedConfig = config
        var det = config.detection ?? DetectionConfig()
        let detName = det.modelPath ?? "det.mnn"
        det.modelPath = resolveModelPath(detName) ?? detName
        resolvedConfig.detection = det

        var rec = config.recognition ?? RecognitionConfig()
        let recName = rec.modelPath ?? "rec.mnn"
        let dictName = rec.dictPath ?? "dict.txt"
        rec.modelPath = resolveModelPath(recName) ?? recName
        rec.dictPath = resolveModelPath(dictName) ?? dictName
        resolvedConfig.recognition = rec

        if var cls = config.classification, let path = cls.modelPath {
            cls.modelPath = resolveModelPath(path) ?? path
            resolvedConfig.classification = cls
        }
        if var orient = config.orientation, let path = orient.modelPath {
            orient.modelPath = resolveModelPath(path) ?? path
            resolvedConfig.orientation = orient
        }
        if var unwarp = config.unwarp, let path = unwarp.modelPath {
            unwarp.modelPath = resolveModelPath(path) ?? path
            resolvedConfig.unwarp = unwarp
        }

        guard let data = try? JSONEncoder().encode(resolvedConfig),
              let json = String(data: data, encoding: .utf8),
              let handle = rocr_initialize(json) else {
            throw RustOError.initializationFailed
        }
        return RustO(handle: handle)
    }

    public func detectText(_ source: ImageSource, options: OcrRunOptions = OcrRunOptions()) throws -> DetectTextResult {
        guard let handle else { throw RustOError.invalidHandle }
        try validate(options)
        guard let data = try? JSONEncoder().encode(options),
              let json = String(data: data, encoding: .utf8) else {
            throw RustOError.serializationFailed
        }
        switch (source, options.output) {
        case let (.uri(value), .lines), let (.uri(value), .words):
            return .structured(try structuredFile(handle, path: resolveFilePath(value), options: json))
        case let (.bytes(data), .lines), let (.bytes(data), .words):
            return .structured(try structuredData(handle, data: data, options: json))
        case let (.uri(value), .spatial):
            guard let text = rocr_detect_text_file_spatial(handle, resolveFilePath(value), json) else {
                throw RustOError.recognitionFailed(-1)
            }
            defer { rocr_free_string(text) }
            return .spatial(String(cString: text))
        case let (.bytes(data), .spatial):
            guard !data.isEmpty else { throw RustOError.invalidPath }
            let text = data.withUnsafeBytes { bytes in
                rocr_detect_text_data_spatial(handle, bytes.baseAddress, data.count, json)
            }
            guard let text else { throw RustOError.recognitionFailed(-1) }
            defer { rocr_free_string(text) }
            return .spatial(String(cString: text))
        }
    }

    private func structuredFile(_ handle: OpaquePointer, path: String, options: String) throws -> [TextResult] {
        var results: UnsafeMutablePointer<CTextResult>?
        var count = 0
        let status = rocr_detect_text_file(handle, path, options, &results, &count)
        guard status == 0 else { throw RustOError.recognitionFailed(status) }
        return try convertResults(results, count: count)
    }

    private func structuredData(_ handle: OpaquePointer, data: Data, options: String) throws -> [TextResult] {
        guard !data.isEmpty else { throw RustOError.invalidPath }
        var results: UnsafeMutablePointer<CTextResult>?
        var count = 0
        let status = data.withUnsafeBytes { bytes in
            rocr_detect_text_data(handle, bytes.baseAddress, data.count, options, &results, &count)
        }
        guard status == 0 else { throw RustOError.recognitionFailed(status) }
        return try convertResults(results, count: count)
    }

    private func convertResults(_ results: UnsafeMutablePointer<CTextResult>?, count: Int) throws -> [TextResult] {
        guard let results else {
            if count == 0 { return [] }
            throw RustOError.recognitionFailed(-1)
        }
        defer { rocr_free_results(results, count) }
        return (0..<count).map { index in
            let result = results[index]
            let points = [
                Point2D(x: result.box_x1, y: result.box_y1),
                Point2D(x: result.box_x2, y: result.box_y2),
                Point2D(x: result.box_x3, y: result.box_y3),
                Point2D(x: result.box_x4, y: result.box_y4),
            ]
            return TextResult(
                text: String(cString: result.text), score: result.score, boxPoints: points,
                frame: Frame(width: result.frame_width, height: result.frame_height, top: result.frame_top, left: result.frame_left)
            )
        }
    }

    private func validate(_ options: OcrRunOptions) throws {
        let validThreshold = { (value: Float?) in value == nil || (value!.isFinite && value! >= 0) }
        guard validThreshold(options.lineYThreshold), validThreshold(options.wordXThreshold),
              options.textScore == nil || (options.textScore!.isFinite && (0...1).contains(options.textScore!)) else {
            throw RustOError.serializationFailed
        }
        guard let preprocessing = options.preprocessing else { return }
        func positive(_ value: Float?) -> Bool { value == nil || (value!.isFinite && value! > 0) }
        guard positive(preprocessing.minHeight), positive(preprocessing.maxSideLen), positive(preprocessing.minSideLen),
              preprocessing.widthHeightRatio == nil || (preprocessing.widthHeightRatio!.isFinite && (preprocessing.widthHeightRatio! > 0 || preprocessing.widthHeightRatio! == -1)),
              preprocessing.minSideLen == nil || preprocessing.maxSideLen == nil || preprocessing.minSideLen! <= preprocessing.maxSideLen! else { throw RustOError.serializationFailed }
        guard let detection = preprocessing.detection else { return }
        guard detection.limitSideLen == nil || (detection.limitSideLen! > 0 && detection.limitSideLen! <= 32767),
              detection.limitType == nil || ["min", "max"].contains(detection.limitType!),
              detection.mean == nil || (detection.mean!.count == 3 && detection.mean!.allSatisfy { $0.isFinite }),
              detection.std == nil || (detection.std!.count == 3 && detection.std!.allSatisfy { $0.isFinite && $0 != 0 }) else { throw RustOError.serializationFailed }
        guard let postprocess = detection.postprocess else { return }
        guard postprocess.threshold == nil || (postprocess.threshold!.isFinite && (0...1).contains(postprocess.threshold!)),
              postprocess.boxThreshold == nil || (postprocess.boxThreshold!.isFinite && (0...1).contains(postprocess.boxThreshold!)),
              postprocess.maxCandidates == nil || postprocess.maxCandidates! >= 1,
              positive(postprocess.unclipRatio) else { throw RustOError.serializationFailed }
    }

    private static func resolveModelPath(_ filename: String) -> String? {
        if filename.hasPrefix("/") && FileManager.default.fileExists(atPath: filename) { return filename }
        let name = filename.replacingOccurrences(of: ".mnn", with: "").replacingOccurrences(of: ".txt", with: "")
        let ext = String(filename.split(separator: ".").last ?? "")
        let bundleNames = ["RustOModels", "RustoModels", "RustOModels_PPOCRv6_Tiny", "RustOModels_PPOCRv6_Small", "RustOModels_PPOCRv6_Medium", "RustOModels_PPOCRv5_Mobile", "RustOModels_PPOCRv5_Server", "RustOModels_PPOCRv4_Mobile", "RustOModels_PPOCRv4_Server"]
        for bundleName in bundleNames {
            if let path = Bundle.main.path(forResource: bundleName, ofType: "bundle"),
               let bundle = Bundle(path: path), let file = bundle.path(forResource: name, ofType: ext) { return file }
        }
        if let file = Bundle.main.path(forResource: filename, ofType: nil) { return file }
        let documents = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
        let file = (documents as NSString).appendingPathComponent(filename)
        return FileManager.default.fileExists(atPath: file) ? file : nil
    }

    private func resolveFilePath(_ value: String) -> String {
        var path = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if path.hasPrefix("file://"), let url = URL(string: path) { path = url.path }
        else if path.hasPrefix("file:") { path = String(path.dropFirst(5)) }
        return path.removingPercentEncoding ?? path
    }

    deinit { if let handle { rocr_free(handle) } }
}

// C API bridge
struct CTextResult {
    let text: UnsafeMutablePointer<CChar>
    let score: Float
    let box_x1, box_y1: Float
    let box_x2, box_y2: Float
    let box_x3, box_y3: Float
    let box_x4, box_y4: Float
    let frame_width, frame_height: Float
    let frame_top, frame_left: Float
}

@_silgen_name("rocr_initialize")
func rocr_initialize(_ configJson: String) -> OpaquePointer?

@_silgen_name("rocr_detect_text_file")
func rocr_detect_text_file(_ handle: OpaquePointer, _ imagePath: String, _ options: String, _ results: UnsafeMutablePointer<UnsafeMutablePointer<CTextResult>?>, _ count: UnsafeMutablePointer<Int>) -> Int32

@_silgen_name("rocr_detect_text_data")
func rocr_detect_text_data(_ handle: OpaquePointer, _ data: UnsafeRawPointer?, _ length: Int, _ options: String, _ results: UnsafeMutablePointer<UnsafeMutablePointer<CTextResult>?>, _ count: UnsafeMutablePointer<Int>) -> Int32

@_silgen_name("rocr_detect_text_file_spatial")
func rocr_detect_text_file_spatial(_ handle: OpaquePointer, _ imagePath: String, _ options: String) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_detect_text_data_spatial")
func rocr_detect_text_data_spatial(_ handle: OpaquePointer, _ data: UnsafeRawPointer?, _ length: Int, _ options: String) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_free_results")
func rocr_free_results(_ results: UnsafeMutablePointer<CTextResult>, _ count: Int)

@_silgen_name("rocr_free_string")
func rocr_free_string(_ str: UnsafeMutablePointer<CChar>)

@_silgen_name("rocr_free")
func rocr_free(_ handle: OpaquePointer)

@_silgen_name("rocr_version")
func rocr_version() -> UnsafePointer<CChar>?
