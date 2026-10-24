import Foundation

public typealias RustOError = RustoBindingError

extension RustO {
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

    public static func resolveFilePath(_ value: String) -> String {
        var path = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if path.hasPrefix("file://"), let url = URL(string: path) { path = url.path }
        else if path.hasPrefix("file:") { path = String(path.dropFirst(5)) }
        return path.removingPercentEncoding ?? path
    }
}

extension InitializeConfig {
    public init(
        template: String? = nil,
        detection: DetectionConfig? = nil,
        recognition: RecognitionConfig? = nil,
        classification: ClassificationConfig? = nil,
        orientation: OrientationConfig? = nil,
        unwarp: UnwarpConfig? = nil
    ) {
        self.init(
            template: template,
            detection: detection,
            recognition: recognition,
            classification: classification,
            orientation: orientation,
            unwarp: unwarp
        )
    }
}

extension DetectionConfig {
    public init(
        enabled: Bool? = nil,
        modelPath: String? = nil,
        thresh: Float? = nil,
        boxThresh: Float? = nil,
        unclipRatio: Float? = nil,
        limitSideLen: Int32? = nil,
        limitType: String? = nil,
        useDilation: Bool? = nil
    ) {
        self.init(
            enabled: enabled,
            modelPath: modelPath,
            thresh: thresh,
            boxThresh: boxThresh,
            unclipRatio: unclipRatio,
            limitSideLen: limitSideLen,
            limitType: limitType,
            useDilation: useDilation
        )
    }
}

extension RecognitionConfig {
    public init(
        enabled: Bool? = nil,
        modelPath: String? = nil,
        dictPath: String? = nil
    ) {
        self.init(
            enabled: enabled,
            modelPath: modelPath,
            dictPath: dictPath
        )
    }
}

extension ClassificationConfig {
    public init(
        enabled: Bool? = nil,
        modelPath: String? = nil,
        thresh: Float? = nil
    ) {
        self.init(
            enabled: enabled,
            modelPath: modelPath,
            thresh: thresh
        )
    }
}

extension OrientationConfig {
    public init(
        enabled: Bool? = nil,
        modelPath: String? = nil,
        thresh: Float? = nil
    ) {
        self.init(
            enabled: enabled,
            modelPath: modelPath,
            thresh: thresh
        )
    }
}

extension UnwarpConfig {
    public init(
        enabled: Bool? = nil,
        modelPath: String? = nil
    ) {
        self.init(
            enabled: enabled,
            modelPath: modelPath
        )
    }
}

extension RuntimeOptions {
    public init(
        output: OutputGranularity = .lines,
        lineYThreshold: Float? = nil,
        wordXThreshold: Float? = nil,
        textScore: Float? = nil,
        classification: Bool? = nil,
        orientation: Bool? = nil,
        minHeight: Float? = nil,
        maxSideLen: Float? = nil,
        minSideLen: Float? = nil,
        widthHeightRatio: Float? = nil,
        detection: DetectionRunOptions? = nil,
        postprocess: PostprocessRunOptions? = nil,
        calibration: CalibrationOptions? = nil,
        optimization: OptimizationOptions? = nil
    ) {
        self.init(
            output: output,
            lineYThreshold: lineYThreshold,
            wordXThreshold: wordXThreshold,
            textScore: textScore,
            classification: classification,
            orientation: orientation,
            minHeight: minHeight,
            maxSideLen: maxSideLen,
            minSideLen: minSideLen,
            widthHeightRatio: widthHeightRatio,
            detection: detection,
            postprocess: postprocess,
            calibration: calibration,
            optimization: optimization
        )
    }
}
