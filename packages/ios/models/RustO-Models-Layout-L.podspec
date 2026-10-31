Pod::Spec.new do |s|
  s.name             = 'RustO-Models-Layout-L'
  s.version          = '0.3.2'
  s.summary          = 'Document layout detection model (Large) for RustO! iOS'
  s.description      = 'PP-DocLayout-Plus-L document layout model (~124 MB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = {
    :http => "https://github.com/byrizki/rusto-rs/releases/download/models-rten-v#{s.version}/RustO-Models-Layout-L.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_Layout_L' => ['**/*.{onnx,rten,mnn,txt}']
  }
end

