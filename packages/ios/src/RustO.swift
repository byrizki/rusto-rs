import Foundation

/// Underlying native RustO binding error.
public typealias RustOError = RustoBindingError

/// Runtime OCR options alias.
public typealias RuntimeOptions = OcrRunOptions

extension RustO {
    /// Library version string of the underlying RustO engine.
    public static var libraryVersion: String {
        rustoVersion()
    }

    /// Initialize RustO with automatic model bundle resolution for iOS.
    public static func initializeWithBundledModels(config: InitializeConfig = InitializeConfig()) throws -> RustO {
        var resolvedConfig = config
        var det = config.detection ?? DetectionConfig()
        let detName = det.modelPath ?? "det.rten"
        det.modelPath = resolveModelPath(detName) ?? detName
        resolvedConfig.detection = det

        var rec = config.recognition ?? RecognitionConfig()
        let recName = rec.modelPath ?? "rec.rten"
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

        return try RustO.initialize(config: resolvedConfig)
    }

    /// Resolves model file path from the app main bundle, model resource bundles, or documents directory.
    ///
    /// - Parameter filename: File name or partial path of model asset.
    /// - Returns: Absolute file path if found, or nil.
    public static func resolveModelPath(_ filename: String) -> String? {
        if FileManager.default.fileExists(atPath: filename) { return filename }
        let ext = (filename as NSString).pathExtension
        let name = (filename as NSString).deletingPathExtension
        let candidateExts = [ext, "rten", "onnx", "mnn", "txt"].filter { !$0.isEmpty }
        let bundleNames = [
            "RustO", "Rusto", "RustOModels", "RustoModels",
            "RustOModels_PPOCRv6_Tiny", "RustOModels_PPOCRv6_Small", "RustOModels_PPOCRv6_Medium",
            "RustOModels_PPOCRv5_Mobile", "RustOModels_PPOCRv5_Server",
            "RustOModels_PPOCRv4_Mobile", "RustOModels_PPOCRv4_Server"
        ]
        for bundleName in bundleNames {
            if let path = Bundle.main.path(forResource: bundleName, ofType: "bundle"),
               let bundle = Bundle(path: path) {
                for candidateExt in candidateExts {
                    if let file = bundle.path(forResource: name, ofType: candidateExt) { return file }
                }
            }
        }
        if let file = Bundle.main.path(forResource: filename, ofType: nil) { return file }
        let documents = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
        let file = (documents as NSString).appendingPathComponent(filename)
        return FileManager.default.fileExists(atPath: file) ? file : nil
    }

    /// Resolves a file URI or path into a standard POSIX filesystem path.
    ///
    /// - Parameter value: File path or `file://` URI string.
    /// - Returns: Clean absolute POSIX path.
    public static func resolveFilePath(_ value: String) -> String {
        var path = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if path.hasPrefix("file://"), let url = URL(string: path) { path = url.path }
        else if path.hasPrefix("file:") { path = String(path.dropFirst(5)) }
        return path.removingPercentEncoding ?? path
    }
}

extension InitializeConfig {
    public init() {
        self.init(
            template: nil,
            detection: nil,
            recognition: nil,
            classification: nil,
            orientation: nil,
            unwarp: nil
        )
    }
}

extension DetectionConfig {
    public init() {
        self.init(
            enabled: nil,
            modelPath: nil,
            thresh: nil,
            boxThresh: nil,
            unclipRatio: nil,
            limitSideLen: nil,
            limitType: nil,
            useDilation: nil,
            scoreMode: nil
        )
    }
}

extension RecognitionConfig {
    public init() {
        self.init(
            enabled: nil,
            modelPath: nil,
            dictPath: nil
        )
    }
}

extension ClassificationConfig {
    public init() {
        self.init(
            enabled: nil,
            modelPath: nil,
            thresh: nil
        )
    }
}

extension OrientationConfig {
    public init() {
        self.init(
            enabled: nil,
            modelPath: nil,
            thresh: nil
        )
    }
}

extension UnwarpConfig {
    public init() {
        self.init(
            enabled: nil,
            modelPath: nil
        )
    }
}

extension PostprocessRunOptions {
    public init() {
        self.init(
            threshold: nil,
            boxThreshold: nil,
            maxCandidates: nil,
            unclipRatio: nil,
            useDilation: nil,
            scoreMode: nil
        )
    }
}

extension DetectionRunOptions {
    public init() {
        self.init(
            limitSideLen: nil,
            limitType: nil,
            mean: nil,
            std: nil,
            postprocess: nil
        )
    }
}

extension CalibrationOptions {
    public init() {
        self.init(
            descreenStrength: nil
        )
    }
}

extension OptimizationOptions {
    public init() {
        self.init(
            cropPaddingX: nil,
            cropPaddingY: nil
        )
    }
}

extension OcrRunOptions {
    public init() {
        self.init(
            output: .lines,
            lineYThreshold: nil,
            wordXThreshold: nil,
            textScore: nil,
            classification: nil,
            orientation: nil,
            minHeight: nil,
            maxSideLen: nil,
            minSideLen: nil,
            widthHeightRatio: nil,
            detection: nil,
            postprocess: nil,
            calibration: nil,
            optimization: nil
        )
    }
}
