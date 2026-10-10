Pod::Spec.new do |s|
  s.name             = 'RustO-Models-PPOCRv6-Small'
  s.version          = '0.3.2'
  s.summary          = 'PP-OCRv6 Small pre-trained models for RustO! iOS'
  s.description      = 'Pre-trained PP-OCRv6 Small models (~30 MB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = { 
    :http => "https://github.com/byrizki/rusto-rs/releases/download/models-rten-v#{s.version}/RustO-Models-PPOCRv6-Small.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_PPOCRv6_Small' => ['**/*.{onnx,rten,mnn,txt}']
  }
end
