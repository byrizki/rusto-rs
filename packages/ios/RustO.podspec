Pod::Spec.new do |s|
  s.name             = 'RustO'
  s.version          = '0.2.5'
  s.summary          = 'RustO! - Pure Rust OCR library for iOS'
  s.description      = <<-DESC
    RustO! is a high-performance OCR (Optical Character Recognition) library 
    written in pure Rust, based on RapidOCR and powered by PaddleOCR models 
    with ONNX Runtime inference. This pod provides Swift bindings for iOS.
  DESC

  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = { 
    :http => "https://github.com/byrizki/rusto-rs/releases/download/v#{s.version}/RustO.xcframework.zip"
  }

  s.ios.deployment_target = '12.0'
  s.swift_version = '5.0'

  # Swift source files
  s.source_files = 'src/**/*.swift'
  
  # XCFramework selects exactly one platform slice. Do not glob individual
  # static archives: CocoaPods otherwise adds device and simulator libraries,
  # then rejects their identical libRustoCore.a names.
  # Core archive name differs from pod target libRustO.a on case-insensitive FS.
  s.vendored_frameworks = 'RustO.xcframework'
  
  # Frameworks
  s.frameworks = 'Foundation'
  
  # Build settings
  s.pod_target_xcconfig = {
    'ENABLE_BITCODE' => 'NO'
  }

end
