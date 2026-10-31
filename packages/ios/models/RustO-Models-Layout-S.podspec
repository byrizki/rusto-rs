Pod::Spec.new do |s|
  s.name             = 'RustO-Models-Layout-S'
  s.version          = '0.3.2'
  s.summary          = 'Document layout detection model (Small) for RustO! iOS'
  s.description      = 'PP-DocLayout-S document layout model (~4.7 MB) bundled as a resource bundle for RustO on iOS.'
  s.homepage         = 'https://github.com/byrizki/rusto-rs'
  s.license          = { :type => 'MIT', :text => 'MIT License' }
  s.author           = { 'RustO Contributors' => 'support@rusto.dev' }
  s.source           = {
    :http => "https://github.com/byrizki/rusto-rs/releases/download/models-rten-v#{s.version}/RustO-Models-Layout-S.zip"
  }
  s.ios.deployment_target = '12.0'
  s.resource_bundles = {
    'RustOModels_Layout_S' => ['**/*.{onnx,rten,mnn,txt}']
  }
end

