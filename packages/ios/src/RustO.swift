import Foundation

public struct Point2D {
    public let x: Float
    public let y: Float
    
    public init(x: Float, y: Float) {
        self.x = x
        self.y = y
    }
}

public struct TextResult {
    public let text: String
    public let score: Float
    public let boxPoints: [Point2D]
    
    public init(text: String, score: Float, boxPoints: [Point2D]) {
        self.text = text
        self.score = score
        self.boxPoints = boxPoints
    }
}

public enum RustOError: Error {
    case initializationFailed
    case recognitionFailed(Int32)
    case invalidHandle
    case invalidPath
}

public class RustO {
    private var handle: OpaquePointer?

    public static var version: String {
        guard let versionPtr = rocr_version() else {
            return "unknown"
        }
        return String(cString: versionPtr)
    }

    public init(
        detModelPath: String? = nil,
        recModelPath: String? = nil,
        dictPath: String? = nil
    ) throws {
        // Use default bundled models if not specified
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
        // If absolute path, use it directly
        if filename.hasPrefix("/") && FileManager.default.fileExists(atPath: filename) {
            return filename
        }
        
        // Try RustoModels.bundle first (for bundled models)
        if let bundlePath = Bundle.main.path(forResource: "RustoModels", ofType: "bundle"),
           let bundle = Bundle(path: bundlePath) {
            let name = filename.replacingOccurrences(of: ".mnn", with: "").replacingOccurrences(of: ".txt", with: "")
            let ext = String(filename.split(separator: ".").last ?? "")
            if let filePath = bundle.path(forResource: name, ofType: ext) {
                return filePath
            }
        }
        
        // Try main bundle
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
            return TextResult(
                text: String(cString: result.text),
                score: result.score,
                boxPoints: [
                    Point2D(x: result.box_x1, y: result.box_y1),
                    Point2D(x: result.box_x2, y: result.box_y2),
                    Point2D(x: result.box_x3, y: result.box_y3),
                    Point2D(x: result.box_x4, y: result.box_y4),
                ]
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
            return TextResult(
                text: String(cString: result.text),
                score: result.score,
                boxPoints: [
                    Point2D(x: result.box_x1, y: result.box_y1),
                    Point2D(x: result.box_x2, y: result.box_y2),
                    Point2D(x: result.box_x3, y: result.box_y3),
                    Point2D(x: result.box_x4, y: result.box_y4),
                ]
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
        yThresholdMultiplier: Float = 0.6,
        xThresholdMultiplier: Float = 1.3
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
}

@_silgen_name("rocr_new")
func rocr_new(
    _ detModel: UnsafePointer<CChar>,
    _ recModel: UnsafePointer<CChar>,
    _ dict: UnsafePointer<CChar>
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
