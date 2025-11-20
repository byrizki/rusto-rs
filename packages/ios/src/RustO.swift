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
        guard let versionPtr = rusto_version() else {
            return "unknown"
        }
        return String(cString: versionPtr)
    }

    public init(
        detModelPath: String,
        recModelPath: String,
        dictPath: String
    ) throws {
        handle = rusto_new(
            detModelPath,
            recModelPath,
            dictPath
        )

        guard handle != nil else {
            throw RustOError.initializationFailed
        }
    }

    public func recognizeFile(_ imagePath: String) throws -> [TextResult] {
        guard let handle = handle else {
            throw RustOError.invalidHandle
        }

        var resultsPtr: UnsafeMutablePointer<CTextResult>?
        var count: Int = 0

        let status = rusto_ocr_file(handle, imagePath, &resultsPtr, &count)
        guard status == 0, let results = resultsPtr else {
            throw RustOError.recognitionFailed(status)
        }

        defer { rusto_free_results(results, count) }

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
            rusto_ocr_data(
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

        defer { rusto_free_results(results, count) }

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

    deinit {
        if let h = handle {
            rusto_free(h)
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

@_silgen_name("rusto_new")
func rusto_new(
    _ detModel: UnsafePointer<CChar>,
    _ recModel: UnsafePointer<CChar>,
    _ dict: UnsafePointer<CChar>
) -> OpaquePointer?

@_silgen_name("rusto_ocr_file")
func rusto_ocr_file(
    _ handle: OpaquePointer,
    _ imagePath: UnsafePointer<CChar>,
    _ resultsOut: UnsafeMutablePointer<UnsafeMutablePointer<CTextResult>?>,
    _ countOut: UnsafeMutablePointer<Int>
) -> Int32

@_silgen_name("rusto_ocr_data")
func rusto_ocr_data(
    _ handle: OpaquePointer,
    _ imageData: UnsafePointer<UInt8>,
    _ imageLen: Int,
    _ resultsOut: UnsafeMutablePointer<UnsafeMutablePointer<CTextResult>?>,
    _ countOut: UnsafeMutablePointer<Int>
) -> Int32

@_silgen_name("rusto_free_results")
func rusto_free_results(_ results: UnsafeMutablePointer<CTextResult>, _ count: Int)

@_silgen_name("rusto_free")
func rusto_free(_ handle: OpaquePointer)

@_silgen_name("rusto_version")
func rusto_version() -> UnsafePointer<CChar>?
