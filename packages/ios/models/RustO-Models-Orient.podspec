Pod::Spec.new do |s|
  s.name             = 'RustO-Models-Orient'
  s.version          = '0.3.2'
  s.summary          = 'Document orientation classifier model for RustO! iOS'
  s.description      = 'PP-LCNet document orientation classifier (~6.4 MB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = {
    :http => "https://github.com/byrizki/rusto-rs/releases/download/models-rten-v#{s.version}/RustO-Models-Orient.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_Orient' => ['**/*.{onnx,rten,mnn,txt}']
  }
end

