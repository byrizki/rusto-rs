import Foundation

/// Underlying native RustO binding error.
public typealias RustOError = RustoBindingError

/// Runtime OCR options alias.
public typealias RuntimeOptions = OcrRunOptions

private let globalDownloadModel = downloadModel

extension RustO {
    /// Library version string of the underlying RustO engine.
    public static var libraryVersion: String {
        rustoVersion()
    }

    /// Returns the engine library version string.
    public static func version() -> String {
        rustoVersion()
    }

    /// Checks if the engine is ready for inference.
    public func isReady() -> Bool {
        return true
    }

    /// Returns the engine library version string.
    public func version() -> String {
        return rustoVersion()
    }

    /// Downloads model assets for a specific OCR pipeline ahead of time.
    public static func downloadModel(model: String? = nil, lang: String? = nil, cacheDir: String? = nil) throws -> [String] {
        try globalDownloadModel(model, lang, cacheDir)
    }


    /// Initialize RustO with zero configuration (uses bundled models if present, otherwise auto-downloads IndraOCR into the app cache).
    public static func initialize() throws -> RustO {
        try initializeWithBundledModels()
    }

    /// Initialize RustO with automatic model bundle resolution for iOS.
    public static func initializeWithBundledModels(config: InitializeConfig = InitializeConfig()) throws -> RustO {
        var resolvedConfig = config
        if resolvedConfig.cacheDir == nil {
            let caches = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true).first ?? NSTemporaryDirectory()
            resolvedConfig.cacheDir = (caches as NSString).appendingPathComponent("rusto/models")
        }

        let hasModelDir = config.modelPath != nil || config.model != nil
        let bundledDet = resolveModelPath("det.rten")
        let bundledRec = resolveModelPath("rec.rten")
        let hasBundledDefaultModels = !hasModelDir && config.detection == nil && config.recognition == nil && bundledDet != nil && bundledRec != nil

        if config.detection != nil || hasBundledDefaultModels {
            var det = config.detection ?? DetectionConfig()
            let detName = det.modelPath ?? "det.rten"
            det.modelPath = resolveModelPath(detName) ?? detName
            resolvedConfig.detection = det
        }

        if config.recognition != nil || hasBundledDefaultModels {
            var rec = config.recognition ?? RecognitionConfig()
            let recName = rec.modelPath ?? "rec.rten"
            let dictName = rec.dictPath ?? "dict.txt"
            rec.modelPath = resolveModelPath(recName) ?? recName
            rec.dictPath = resolveModelPath(dictName) ?? dictName
            resolvedConfig.recognition = rec
        }

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
        if var layout = config.layout, let path = layout.modelPath {
            layout.modelPath = resolveModelPath(path) ?? path
            resolvedConfig.layout = layout
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

    /// Creates a fluent DetectTextBuilder for text detection and recognition.
    public func detectText() -> DetectTextBuilder {
        return DetectTextBuilder(engine: self)
    }

    /// Creates a fluent DetectTextBuilder for the specified image file path or URI.
    public func detectText(image path: String) -> DetectTextBuilder {
        return DetectTextBuilder(engine: self).image(path)
    }

    /// Creates a fluent DetectTextBuilder for the specified raw image data.
    public func detectText(data: Data) -> DetectTextBuilder {
        return DetectTextBuilder(engine: self).data(data)
    }

    /// Creates a fluent DetectTextBuilder for the specified ImageSource.
    public func detectText(source: ImageSource) -> DetectTextBuilder {
        return DetectTextBuilder(engine: self, source: source)
    }

    /// Creates a fluent DetectTextBuilder for text detection and recognition.
    public func detectTextBuilder() -> DetectTextBuilder {
        return DetectTextBuilder(engine: self)
    }
}

/// Fluent builder for chained text detection configuration and execution.
public final class DetectTextBuilder {
    private let engine: RustO
    private var source: ImageSource?
    private var rgbInput: (pixels: Data, width: UInt32, height: UInt32, stride: UInt64)?
    private var options: OcrRunOptions = OcrRunOptions()

    public init(engine: RustO, source: ImageSource? = nil) {
        self.engine = engine
        self.source = source
    }

    @discardableResult
    public func image(_ path: String) -> DetectTextBuilder {
        self.source = .path(path: RustO.resolveFilePath(path))
        self.rgbInput = nil
        return self
    }

    @discardableResult
    public func data(_ bytes: Data) -> DetectTextBuilder {
        self.source = .bytes(data: bytes)
        self.rgbInput = nil
        return self
    }

    @discardableResult
    public func rgb(_ pixels: Data, width: UInt32, height: UInt32, stride: UInt64? = nil) -> DetectTextBuilder {
        self.rgbInput = (pixels: pixels, width: width, height: height, stride: stride ?? UInt64(width) * 3)
        self.source = nil
        return self
    }

    @discardableResult
    public func source(_ source: ImageSource) -> DetectTextBuilder {
        self.source = source
        self.rgbInput = nil
        return self
    }

    @discardableResult
    public func withOptions(_ options: OcrRunOptions) -> DetectTextBuilder {
        self.options = options
        return self
    }

    @discardableResult
    public func withAutoUnwarp(_ enabled: Bool = true) -> DetectTextBuilder {
        self.options.autoUnwarp = enabled
        return self
    }

    @discardableResult
    public func withLayoutReading(_ enabled: Bool = true) -> DetectTextBuilder {
        self.options.layoutReading = enabled
        return self
    }

    @discardableResult
    public func withOrientationDetection(_ enabled: Bool = true) -> DetectTextBuilder {
        self.options.orientationDetection = enabled
        return self
    }

    @discardableResult
    public func withLineClassification(_ enabled: Bool = true) -> DetectTextBuilder {
        self.options.lineClassification = enabled
        return self
    }

    @discardableResult
    public func withPreDetection(_ preDetection: PreDetectionOptions) -> DetectTextBuilder {
        self.options.preDetection = preDetection
        return self
    }

    @discardableResult
    public func withPreDetection(_ configure: (inout PreDetectionOptions) -> Void) -> DetectTextBuilder {
        var opts = self.options.preDetection ?? PreDetectionOptions()
        configure(&opts)
        self.options.preDetection = opts
        return self
    }

    @discardableResult
    public func withDetection(_ detection: DetectionRunOptions) -> DetectTextBuilder {
        self.options.detection = detection
        return self
    }

    @discardableResult
    public func withDetection(_ configure: (inout DetectionRunOptions) -> Void) -> DetectTextBuilder {
        var opts = self.options.detection ?? DetectionRunOptions()
        configure(&opts)
        self.options.detection = opts
        return self
    }

    @discardableResult
    public func withPostDetection(_ postDetection: PostDetectionOptions) -> DetectTextBuilder {
        self.options.postDetection = postDetection
        return self
    }

    @discardableResult
    public func withPostDetection(_ configure: (inout PostDetectionOptions) -> Void) -> DetectTextBuilder {
        var opts = self.options.postDetection ?? PostDetectionOptions()
        configure(&opts)
        self.options.postDetection = opts
        return self
    }

    @discardableResult
    public func withPreRecognition(_ preRecognition: PreRecognitionOptions) -> DetectTextBuilder {
        self.options.preRecognition = preRecognition
        return self
    }

    @discardableResult
    public func withPreRecognition(_ configure: (inout PreRecognitionOptions) -> Void) -> DetectTextBuilder {
        var opts = self.options.preRecognition ?? PreRecognitionOptions()
        configure(&opts)
        self.options.preRecognition = opts
        return self
    }

    @discardableResult
    public func withRecognition(_ recognition: RecognitionRunOptions) -> DetectTextBuilder {
        self.options.recognition = recognition
        return self
    }

    @discardableResult
    public func withRecognition(_ configure: (inout RecognitionRunOptions) -> Void) -> DetectTextBuilder {
        var opts = self.options.recognition ?? RecognitionRunOptions()
        configure(&opts)
        self.options.recognition = opts
        return self
    }

    @discardableResult
    public func withPostRecognition(_ postRecognition: PostRecognitionOptions) -> DetectTextBuilder {
        self.options.postRecognition = postRecognition
        return self
    }

    @discardableResult
    public func withPostRecognition(_ configure: (inout PostRecognitionOptions) -> Void) -> DetectTextBuilder {
        var opts = self.options.postRecognition ?? PostRecognitionOptions()
        configure(&opts)
        self.options.postRecognition = opts
        return self
    }

    @discardableResult
    public func withOutput(_ granularity: OutputGranularity) -> DetectTextBuilder {
        self.options.output = granularity
        return self
    }

    @discardableResult
    public func asLines() -> DetectTextBuilder {
        return withOutput(.lines)
    }

    @discardableResult
    public func asWords() -> DetectTextBuilder {
        return withOutput(.words)
    }

    @discardableResult
    public func asSpatial() -> DetectTextBuilder {
        return withOutput(.spatial)
    }

    @discardableResult
    public func asTsv() -> DetectTextBuilder {
        return withOutput(.tsv)
    }

    public func run() throws -> DetectTextResult {
        if let rgb = self.rgbInput {
            return try engine.detectTextRgb(
                pixels: rgb.pixels,
                width: rgb.width,
                height: rgb.height,
                stride: rgb.stride,
                options: options
            )
        }
        guard let source = self.source else {
            throw RustoBindingError.InvalidInput(msg: "Image source must be specified before calling run().")
        }
        return try engine.detectText(source: source, options: options)
    }
}

extension InitializeConfig {
    public init() {
        self.init(
            model: nil,
            modelPath: nil,
            template: nil,
            lang: nil,
            cacheDir: nil,
            enableReadingOrder: nil,
            layoutVariant: nil,
            detection: nil,
            recognition: nil,
            classification: nil,
            orientation: nil,
            unwarp: nil,
            layout: nil
        )
    }

    public static func forModel(model: String? = "indraocr", lang: String? = nil, modelPath: String? = nil) -> InitializeConfig {
        createInitializeConfigForModel(model: model, lang: lang, modelPath: modelPath)
    }

    public static func indra(detModelPath: String? = nil, recModelPath: String? = nil, dictPath: String? = nil) -> InitializeConfig {
        if let det = detModelPath, let rec = recModelPath, let dict = dictPath {
            return createInitializeConfigIndra(detModelPath: det, recModelPath: rec, dictPath: dict)
        }
        var cfg = InitializeConfig()
        cfg.template = "indra"
        if detModelPath != nil {
            var det = DetectionConfig()
            det.enabled = true
            det.modelPath = detModelPath
            cfg.detection = det
        }
        if recModelPath != nil || dictPath != nil {
            var rec = RecognitionConfig()
            rec.enabled = true
            rec.modelPath = recModelPath
            rec.dictPath = dictPath
            rec.recBatchNum = 1
            cfg.recognition = rec
        }
        return cfg
    }

    public static func ppv6(detModelPath: String? = nil, recModelPath: String? = nil, dictPath: String? = nil) -> InitializeConfig {
        if let det = detModelPath, let rec = recModelPath, let dict = dictPath {
            return createInitializeConfigPpv6(detModelPath: det, recModelPath: rec, dictPath: dict)
        }
        return preset(template: "ppv6", detModelPath: detModelPath, recModelPath: recModelPath, dictPath: dictPath)
    }

    public static func ppv5(detModelPath: String? = nil, recModelPath: String? = nil, dictPath: String? = nil) -> InitializeConfig {
        preset(template: "ppv5", detModelPath: detModelPath, recModelPath: recModelPath, dictPath: dictPath)
    }

    public static func ppv4(detModelPath: String? = nil, recModelPath: String? = nil, dictPath: String? = nil) -> InitializeConfig {
        preset(template: "ppv4", detModelPath: detModelPath, recModelPath: recModelPath, dictPath: dictPath)
    }

    public static func ppv3(detModelPath: String? = nil, recModelPath: String? = nil, dictPath: String? = nil) -> InitializeConfig {
        preset(template: "ppv3", detModelPath: detModelPath, recModelPath: recModelPath, dictPath: dictPath)
    }

    private static func preset(template: String, detModelPath: String?, recModelPath: String?, dictPath: String?) -> InitializeConfig {
        var cfg = InitializeConfig()
        cfg.template = template
        if detModelPath != nil {
            var det = DetectionConfig()
            det.enabled = true
            det.modelPath = detModelPath
            cfg.detection = det
        }
        if recModelPath != nil || dictPath != nil {
            var rec = RecognitionConfig()
            rec.enabled = true
            rec.modelPath = recModelPath
            rec.dictPath = dictPath
            cfg.recognition = rec
        }
        return cfg
    }
}

extension LayoutConfig {
    public init() {
        self.init(
            enabled: nil,
            modelPath: nil,
            confThresh: nil,
            targetSize: nil,
            targetHeight: nil,
            targetWidth: nil
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
            dictPath: nil,
            recBatchNum: nil
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

extension PreDetectionOptions {
    public init() {
        self.init(
            descreenStrength: nil,
            targetMaxSide: nil,
            cropPaddingX: nil,
            cropPaddingY: nil
        )
    }
}

extension DetectionRunOptions {
    public init() {
        self.init(
            limitSideLen: nil,
            limitType: nil,
            mean: nil,
            std: nil
        )
    }
}

extension PostDetectionOptions {
    public init() {
        self.init(
            threshold: nil,
            boxThreshold: nil,
            unclipRatio: nil,
            useDilation: nil,
            scoreMode: nil,
            maxCandidates: nil,
            minHeight: nil,
            minSideLen: nil,
            maxSideLen: nil,
            widthHeightRatio: nil
        )
    }
}

extension PreRecognitionOptions {
    public init() {
        self.init(
            lineYThreshold: nil,
            wordXThreshold: nil
        )
    }
}

extension RecognitionRunOptions {
    public init() {
        self.init(
            batchSize: nil
        )
    }
}

extension PostRecognitionOptions {
    public init() {
        self.init(
            scoreThreshold: nil
        )
    }
}

extension OcrRunOptions {
    public init() {
        self.init(
            output: .lines,
            autoUnwarp: nil,
            layoutReading: nil,
            orientationDetection: nil,
            lineClassification: nil,
            preDetection: nil,
            detection: nil,
            postDetection: nil,
            preRecognition: nil,
            recognition: nil,
            postRecognition: nil
        )
    }
}
