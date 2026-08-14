# RustO React Native legacy-architecture consumer

React Native **0.75.5** consumer. `android/gradle.properties` pins
`newArchEnabled=false`. It validates bridge-module compatibility with current
RustO release artifacts.

```bash
yarn install --ignore-scripts --non-interactive
yarn test
yarn typecheck
(cd android && ./gradlew :app:assembleDebug)
```

Release CI downloads current AAR, XCFramework, and npm tarball artifacts before
compiling this app. For local iOS testing, place built `RustO.xcframework` in
`../../packages/ios/`, then:

```bash
(cd ios && pod install)
xcodebuild -workspace ios/RustOLegacyExample.xcworkspace -scheme RustOLegacyExample \
  -configuration Debug -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 15' build
```

App does not invoke real OCR automatically. Models/native runtime required for
runtime inference.
