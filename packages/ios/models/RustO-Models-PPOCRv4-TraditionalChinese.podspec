Pod::Spec.new do |s|
  s.name             = 'RustO-Models-PPOCRv4-TraditionalChinese'
  s.version          = '0.2.0'
  s.summary          = 'PP-OCRv4 Traditional Chinese script recognition model for RustO! iOS'
  s.description      = 'Pre-trained PP-OCRv4 Traditional Chinese script MNN recognition model bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = {
    :http => "https://github.com/byrizki/rusto-rs/releases/download/v#{s.version}/RustO-Models-PPOCRv4-TraditionalChinese.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_PPOCRv4_TraditionalChinese' => ['**/*.{mnn,txt}']
  }
end
