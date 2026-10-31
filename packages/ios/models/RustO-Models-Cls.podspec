Pod::Spec.new do |s|
  s.name             = 'RustO-Models-Cls'
  s.version          = '0.3.2'
  s.summary          = 'Text-line angle classifier model for RustO! iOS'
  s.description      = 'PP-OCRv4 text-line angle classifier (~585 KB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = {
    :http => "https://github.com/byrizki/rusto-rs/releases/download/models-rten-v#{s.version}/RustO-Models-Cls.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_Cls' => ['**/*.{onnx,rten,mnn,txt}']
  }
end

