Pod::Spec.new do |s|
  s.name             = 'RustO-Models-PPOCRv6-Tiny-INT8'
  s.version          = '0.3.1'
  s.summary          = 'PP-OCRv6 Tiny INT8 pre-trained models for RustO! iOS'
  s.description      = 'Pre-trained PP-OCRv6 Tiny INT8 quantized models (~3.4 MB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = { 
    :http => "https://github.com/byrizki/rusto-rs/releases/download/models-rten-v#{s.version}/RustO-Models-PPOCRv6-Tiny-INT8.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_PPOCRv6_Tiny_INT8' => ['**/*.{onnx,rten,mnn,txt}']
  }
end
