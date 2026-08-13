Pod::Spec.new do |s|
  s.name             = 'RustO-Models-PPOCRv5-EastSlavic'
  s.version          = '0.2.1'
  s.summary          = 'PP-OCRv5 East Slavic recognition model for RustO! iOS'
  s.description      = 'Pre-trained PP-OCRv5 East Slavic MNN recognition model bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = {
    :http => "https://github.com/byrizki/rusto-rs/releases/download/v#{s.version}/RustO-Models-PPOCRv5-EastSlavic.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_PPOCRv5_EastSlavic' => ['**/*.{mnn,txt}']
  }
end
