Pod::Spec.new do |s|
  s.name             = 'RustO'
  s.version          = '0.1.0'
  s.summary          = 'RustO! - Pure Rust OCR library for iOS'
  s.description      = <<-DESC
    RustO! is a high-performance OCR (Optical Character Recognition) library 
    written in pure Rust, based on RapidOCR and powered by PaddleOCR models 
    with ONNX Runtime inference. This pod provides Swift bindings for iOS.
  DESC

  s.homepage         = 'https://github.com/yourusername/rusto-rs'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = { 
    :http => "https://github.com/yourusername/rusto-rs/releases/download/v#{s.version}/RustO.xcframework.zip"
  }

  s.ios.deployment_target = '12.0'
  s.swift_version = '5.0'

  # Swift source files
  s.source_files = 'packages/ios/src/**/*.swift'
  
  # Use prebuilt XCFramework (multi-arch support: arm64 device + arm64/x86_64 simulator)
  s.vendored_frameworks = 'RustO.xcframework'
  
  # Model files and dict files as resources
  s.resources = 'packages/ios/Resources/**/*.{mnn,txt}'
  s.resource_bundles = {
    'RustOModels' => ['packages/ios/Resources/**/*.{mnn,txt}']
  }
  
  # Frameworks
  s.frameworks = 'Foundation'
  
  # Build settings
  s.pod_target_xcconfig = {
    'ENABLE_BITCODE' => 'NO'
  }

end
