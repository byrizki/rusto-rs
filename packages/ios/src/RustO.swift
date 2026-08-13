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

public struct RustOConfig: Codable {
    public var template: String = "ppv6"
    public var detModelPath: String = "det.mnn"
    public var recModelPath: String = "rec.mnn"
    public var dictPath: String = "dict.txt"
    public var clsModelPath: String? = nil
    public var orientModelPath: String? = nil
    public var unwarpModelPath: String? = nil
    public var orientThreshold: Float? = nil
    public var clsThreshold: Float? = nil
    public var textScore: Float = 0.5
    public var detThresh: Float = 0.3
    public var detBoxThresh: Float = 0.6
    public var limitSideLen: Int = 736
    public var limitType: String = "min"
    public var unclipRatio: Float = 2.0
    public var useDilation: Bool = true
    public var useDet: Bool = true
    public var useRec: Bool = true
    public var useCls: Bool = false
    public var useOrient: Bool = false
    public var useUnwarp: Bool = false
    public var debugImages: Bool = false
    public var minHeight: Float = 30.0
    public var maxSideLen: Float = 2000.0
    public var minSideLen: Float = 30.0
    public var returnWordBox: Bool = false
    public var returnSingleCharBox: Bool = false
    public var yThresholdMultiplier: Float? = nil
    public var xThresholdMultiplier: Float? = nil

    public init(
        detModelPath: String = "det.mnn",
        recModelPath: String = "rec.mnn",
        dictPath: String = "dict.txt",
        template: String = "ppv6"
    ) {
        self.template = template
        self.detModelPath = detModelPath
        self.recModelPath = recModelPath
        self.dictPath = dictPath
    }

    public static func ppv6(det: String = "det.mnn", rec: String = "rec.mnn", dict: String = "dict.txt") -> RustOConfig {
        RustOConfig(detModelPath: det, recModelPath: rec, dictPath: dict, template: "ppv6")
    }

    public static func ppv5(det: String = "det.mnn", rec: String = "rec.mnn", dict: String = "dict.txt") -> RustOConfig {
        RustOConfig(detModelPath: det, recModelPath: rec, dictPath: dict, template: "ppv5")
    }

    public static func ppv4(det: String = "det.mnn", rec: String = "rec.mnn", dict: String = "dict.txt") -> RustOConfig {
        var cfg = RustOConfig(detModelPath: det, recModelPath: rec, dictPath: dict, template: "ppv4")
        cfg.limitSideLen = 960
        cfg.limitType = "max"
        cfg.unclipRatio = 1.5
        cfg.useDilation = false
        return cfg
    }

    public static func ppv3(det: String = "det.mnn", rec: String = "rec.mnn", dict: String = "dict.txt") -> RustOConfig {
        var cfg = RustOConfig(detModelPath: det, recModelPath: rec, dictPath: dict, template: "ppv3")
        cfg.limitSideLen = 960
        cfg.limitType = "max"
        cfg.unclipRatio = 1.5
        cfg.useDilation = false
        return cfg
    }
}

public enum RustOError: Error {
    case initializationFailed
    case recognitionFailed(Int32)
    case invalidHandle
    case invalidPath
    case serializationFailed
}

public class RustO {
    private var handle: OpaquePointer?

    public static var version: String {
        guard let versionPtr = rocr_version() else {
            return "unknown"
        }
        return String(cString: versionPtr)
    }

    public init(config: RustOConfig) throws {
        var resolvedConfig = config
        resolvedConfig.detModelPath = resolveModelPath(config.detModelPath) ?? config.detModelPath
        resolvedConfig.recModelPath = resolveModelPath(config.recModelPath) ?? config.recModelPath
        resolvedConfig.dictPath = resolveModelPath(config.dictPath) ?? config.dictPath

        if let cls = config.clsModelPath {
            resolvedConfig.clsModelPath = resolveModelPath(cls) ?? cls
        }
        if let orient = config.orientModelPath {
            resolvedConfig.orientModelPath = resolveModelPath(orient) ?? orient
        }
        if let unwarp = config.unwarpModelPath {
            resolvedConfig.unwarpModelPath = resolveModelPath(unwarp) ?? unwarp
        }

        let encoder = JSONEncoder()
        guard let data = try? encoder.encode(resolvedConfig),
              let jsonStr = String(data: data, encoding: .utf8) else {
            throw RustOError.serializationFailed
        }

        handle = rocr_new_with_config(jsonStr)
        guard handle != nil else {
            throw RustOError.initializationFailed
        }
    }

    public init(
        detModelPath: String? = nil,
        recModelPath: String? = nil,
        dictPath: String? = nil
    ) throws {
        let detPath = detModelPath ?? resolveModelPath("det.mnn")
        let recPath = recModelPath ?? resolveModelPath("rec.mnn")
        let dictFile = dictPath ?? resolveModelPath("dict.txt")
        
        guard let detPath = detPath, let recPath = recPath, let dictFile = dictFile else {
            throw RustOError.invalidPath
        }
        
        handle = rocr_new(
            detPath,
            recPath,
            dictFile
        )

        guard handle != nil else {
            throw RustOError.initializationFailed
        }
    }
    
    private func resolveModelPath(_ filename: String) -> String? {
        if filename.hasPrefix("/") && FileManager.default.fileExists(atPath: filename) {
            return filename
        }
        
        let name = filename.replacingOccurrences(of: ".mnn", with: "").replacingOccurrences(of: ".txt", with: "")
        let ext = String(filename.split(separator: ".").last ?? "")
        
        let bundleNames = [
            "RustOModels", "RustoModels",
            "RustOModels_PPOCRv6_Tiny", "RustOModels_PPOCRv6_Small", "RustOModels_PPOCRv6_Medium",
            "RustOModels_PPOCRv5_Mobile", "RustOModels_PPOCRv5_Server",
            "RustOModels_PPOCRv4_Mobile", "RustOModels_PPOCRv4_Server"
        ]
        
        for bName in bundleNames {
            if let bundlePath = Bundle.main.path(forResource: bName, ofType: "bundle"),
               let bundle = Bundle(path: bundlePath),
               let filePath = bundle.path(forResource: name, ofType: ext) {
                return filePath
            }
        }
        
        if let bundlePath = Bundle.main.path(forResource: filename, ofType: nil) {
            return bundlePath
        }
        
        let documentsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
        let filePath = (documentsPath as NSString).appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: filePath) {
            return filePath
        }
        
        return nil
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

    public func recognizeFile(_ imagePath: String) throws -> [TextResult] {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }

        let resolvedPath = resolveFilePath(imagePath)
        var resultsPtr: UnsafeMutablePointer<CTextResult>?
        var count: Int = 0

        let status = rocr_ocr_file(handle, resolvedPath, &resultsPtr, &count)
        guard status == 0, let results = resultsPtr else {
            throw RustOError.recognitionFailed(status)
        }

        defer { rocr_free_results(results, count) }

        return (0..<count).map { i in
            let result = results[i]
            let boxPoints = [
                Point2D(x: result.box_x1, y: result.box_y1),
                Point2D(x: result.box_x2, y: result.box_y2),
                Point2D(x: result.box_x3, y: result.box_y3),
                Point2D(x: result.box_x4, y: result.box_y4),
            ]
            let frame = Frame(
                width: result.frame_width,
                height: result.frame_height,
                top: result.frame_top,
                left: result.frame_left
            )
            return TextResult(
                text: String(cString: result.text),
                score: result.score,
                boxPoints: boxPoints,
                frame: frame
            )
        }
    }

    public func recognize(_ imageData: Data) throws -> [TextResult] {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }

        var resultsPtr: UnsafeMutablePointer<CTextResult>?
        var count: Int = 0

        let status = imageData.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            rocr_ocr_data(
                handle,
                bytes.baseAddress!.assumingMemoryBound(to: UInt8.self),
                bytes.count,
                &resultsPtr,
                &count
            )
        }

        guard status == 0, let results = resultsPtr else {
            throw RustOError.recognitionFailed(status)
        }

        defer { rocr_free_results(results, count) }

        return (0..<count).map { i in
            let result = results[i]
            let boxPoints = [
                Point2D(x: result.box_x1, y: result.box_y1),
                Point2D(x: result.box_x2, y: result.box_y2),
                Point2D(x: result.box_x3, y: result.box_y3),
                Point2D(x: result.box_x4, y: result.box_y4),
            ]
            let frame = Frame(
                width: result.frame_width,
                height: result.frame_height,
                top: result.frame_top,
                left: result.frame_left
            )
            return TextResult(
                text: String(cString: result.text),
                score: result.score,
                boxPoints: boxPoints,
                frame: frame
            )
        }
    }

    public func recognizeFileToRaw(_ imagePath: String) throws -> String {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }
        
        let resolvedPath = resolveFilePath(imagePath)
        var outputPtr: OpaquePointer?
        let status = rocr_ocr_file_with_output(handle, resolvedPath, &outputPtr)
        guard status == 0, let output = outputPtr else {
            throw RustOError.recognitionFailed(status)
        }
        
        defer { rocr_free_output(output) }
        
        guard let strPtr = rocr_output_to_raw(output) else {
            return ""
        }
        defer { rocr_free_string(strPtr) }
        
        return String(cString: strPtr)
    }
    
    public func recognizeFileToCsv(_ imagePath: String) throws -> String {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }
        
        let resolvedPath = resolveFilePath(imagePath)
        var outputPtr: OpaquePointer?
        let status = rocr_ocr_file_with_output(handle, resolvedPath, &outputPtr)
        guard status == 0, let output = outputPtr else {
            throw RustOError.recognitionFailed(status)
        }
        
        defer { rocr_free_output(output) }
        
        guard let strPtr = rocr_output_to_csv(output) else {
            return ""
        }
        defer { rocr_free_string(strPtr) }
        
        return String(cString: strPtr)
    }
    
    public func recognizeFileToTextWithPosition(_ imagePath: String) throws -> String {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }
        
        let resolvedPath = resolveFilePath(imagePath)
        var outputPtr: OpaquePointer?
        let status = rocr_ocr_file_with_output(handle, resolvedPath, &outputPtr)
        guard status == 0, let output = outputPtr else {
            throw RustOError.recognitionFailed(status)
        }
        
        defer { rocr_free_output(output) }
        
        guard let strPtr = rocr_output_to_text_with_position(output) else {
            return ""
        }
        defer { rocr_free_string(strPtr) }
        
        return String(cString: strPtr)
    }
    
    public func recognizeFileToSpatialText(
        _ imagePath: String,
        yThresholdMultiplier: Float = 0.0,
        xThresholdMultiplier: Float = 0.0
    ) throws -> String {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }
        
        let resolvedPath = resolveFilePath(imagePath)
        var outputPtr: OpaquePointer?
        let status = rocr_ocr_file_with_output(handle, resolvedPath, &outputPtr)
        guard status == 0, let output = outputPtr else {
            throw RustOError.recognitionFailed(status)
        }
        
        defer { rocr_free_output(output) }
        
        guard let strPtr = rocr_output_to_spatial_text(output, yThresholdMultiplier, xThresholdMultiplier) else {
            return ""
        }
        defer { rocr_free_string(strPtr) }
        
        return String(cString: strPtr)
    }

    public func recognizeDataToRaw(_ imageData: Data) throws -> String {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }
        var outputPtr: OpaquePointer?
        let status = imageData.withUnsafeBytes { bytes in
            rocr_ocr_data_with_output(handle, bytes.baseAddress!.assumingMemoryBound(to: UInt8.self), bytes.count, &outputPtr)
        }
        guard status == 0, let output = outputPtr else {
            throw RustOError.recognitionFailed(status)
        }
        defer { rocr_free_output(output) }
        guard let strPtr = rocr_output_to_raw(output) else { return "" }
        defer { rocr_free_string(strPtr) }
        return String(cString: strPtr)
    }

    public func recognizeDataToCsv(_ imageData: Data) throws -> String {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }
        var outputPtr: OpaquePointer?
        let status = imageData.withUnsafeBytes { bytes in
            rocr_ocr_data_with_output(handle, bytes.baseAddress!.assumingMemoryBound(to: UInt8.self), bytes.count, &outputPtr)
        }
        guard status == 0, let output = outputPtr else {
            throw RustOError.recognitionFailed(status)
        }
        defer { rocr_free_output(output) }
        guard let strPtr = rocr_output_to_csv(output) else { return "" }
        defer { rocr_free_string(strPtr) }
        return String(cString: strPtr)
    }

    public func recognizeDataToTextWithPosition(_ imageData: Data) throws -> String {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }
        var outputPtr: OpaquePointer?
        let status = imageData.withUnsafeBytes { bytes in
            rocr_ocr_data_with_output(handle, bytes.baseAddress!.assumingMemoryBound(to: UInt8.self), bytes.count, &outputPtr)
        }
        guard status == 0, let output = outputPtr else {
            throw RustOError.recognitionFailed(status)
        }
        defer { rocr_free_output(output) }
        guard let strPtr = rocr_output_to_text_with_position(output) else { return "" }
        defer { rocr_free_string(strPtr) }
        return String(cString: strPtr)
    }

    public func recognizeDataToSpatialText(
        _ imageData: Data,
        yThresholdMultiplier: Float = 0.0,
        xThresholdMultiplier: Float = 0.0
    ) throws -> String {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }
        var outputPtr: OpaquePointer?
        let status = imageData.withUnsafeBytes { bytes in
            rocr_ocr_data_with_output(handle, bytes.baseAddress!.assumingMemoryBound(to: UInt8.self), bytes.count, &outputPtr)
        }
        guard status == 0, let output = outputPtr else {
            throw RustOError.recognitionFailed(status)
        }
        defer { rocr_free_output(output) }
        guard let strPtr = rocr_output_to_spatial_text(output, yThresholdMultiplier, xThresholdMultiplier) else { return "" }
        defer { rocr_free_string(strPtr) }
        return String(cString: strPtr)
    }

    deinit {
        if let h = handle {
            rocr_free(h)
        }
    }
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

@_silgen_name("rocr_new")
func rocr_new(
    _ detModel: UnsafePointer<CChar>,
    _ recModel: UnsafePointer<CChar>,
    _ dict: UnsafePointer<CChar>
) -> OpaquePointer?

@_silgen_name("rocr_new_with_config")
func rocr_new_with_config(
    _ configJson: UnsafePointer<CChar>
) -> OpaquePointer?

@_silgen_name("rocr_ocr_file")
func rocr_ocr_file(
    _ handle: OpaquePointer,
    _ imagePath: UnsafePointer<CChar>,
    _ resultsOut: UnsafeMutablePointer<UnsafeMutablePointer<CTextResult>?>,
    _ countOut: UnsafeMutablePointer<Int>
) -> Int32

@_silgen_name("rocr_ocr_data")
func rocr_ocr_data(
    _ handle: OpaquePointer,
    _ imageData: UnsafePointer<UInt8>,
    _ imageLen: Int,
    _ resultsOut: UnsafeMutablePointer<UnsafeMutablePointer<CTextResult>?>,
    _ countOut: UnsafeMutablePointer<Int>
) -> Int32

@_silgen_name("rocr_free_results")
func rocr_free_results(_ results: UnsafeMutablePointer<CTextResult>, _ count: Int)

@_silgen_name("rocr_free")
func rocr_free(_ handle: OpaquePointer)

@_silgen_name("rocr_ocr_file_with_output")
func rocr_ocr_file_with_output(
    _ handle: OpaquePointer,
    _ imagePath: UnsafePointer<CChar>,
    _ outputOut: UnsafeMutablePointer<OpaquePointer?>
) -> Int32

@_silgen_name("rocr_ocr_data_with_output")
func rocr_ocr_data_with_output(
    _ handle: OpaquePointer,
    _ imageData: UnsafePointer<UInt8>,
    _ imageLen: Int,
    _ outputOut: UnsafeMutablePointer<OpaquePointer?>
) -> Int32

@_silgen_name("rocr_output_to_raw")
func rocr_output_to_raw(_ output: OpaquePointer) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_output_to_csv")
func rocr_output_to_csv(_ output: OpaquePointer) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_output_to_text_with_position")
func rocr_output_to_text_with_position(_ output: OpaquePointer) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_output_to_spatial_text")
func rocr_output_to_spatial_text(
    _ output: OpaquePointer,
    _ yThresholdMultiplier: Float,
    _ xThresholdMultiplier: Float
) -> UnsafeMutablePointer<CChar>?

@_silgen_name("rocr_free_output")
func rocr_free_output(_ output: OpaquePointer)

@_silgen_name("rocr_free_string")
func rocr_free_string(_ str: UnsafeMutablePointer<CChar>)

@_silgen_name("rocr_version")
func rocr_version() -> UnsafePointer<CChar>?
