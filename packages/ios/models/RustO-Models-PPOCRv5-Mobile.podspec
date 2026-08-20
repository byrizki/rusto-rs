Pod::Spec.new do |s|
  s.name             = 'RustO-Models-PPOCRv5-Mobile'
  s.version          = '0.2.4'
  s.summary          = 'PP-OCRv5 Mobile pre-trained models for RustO! iOS'
  s.description      = 'Pre-trained PP-OCRv5 Mobile MNN models (~28 MB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = { 
    :http => "https://github.com/byrizki/rusto-rs/releases/download/v#{s.version}/RustO-Models-PPOCRv5-Mobile.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_PPOCRv5_Mobile' => ['**/*.{mnn,txt}']
  }
end
