Pod::Spec.new do |s|
  s.name             = 'RustO-Models-PPOCRv4-Server'
  s.version          = '0.2.4'
  s.summary          = 'PP-OCRv4 Server pre-trained high-accuracy models for RustO! iOS'
  s.description      = 'Pre-trained PP-OCRv4 Server MNN models with orientation classifier (~300 MB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = { 
    :http => "https://github.com/byrizki/rusto-rs/releases/download/v#{s.version}/RustO-Models-PPOCRv4-Server.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_PPOCRv4_Server' => ['**/*.{mnn,txt}']
  }
end
