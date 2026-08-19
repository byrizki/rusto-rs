Pod::Spec.new do |s|
  s.name             = 'RustO-Models-PPOCRv5-Server'
  s.version          = '0.2.3'
  s.summary          = 'PP-OCRv5 Server pre-trained high-accuracy models for RustO! iOS'
  s.description      = 'Pre-trained PP-OCRv5 Server MNN models (~270 MB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = { 
    :http => "https://github.com/byrizki/rusto-rs/releases/download/v#{s.version}/RustO-Models-PPOCRv5-Server.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_PPOCRv5_Server' => ['**/*.{mnn,txt}']
  }
end
