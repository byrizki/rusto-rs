# RustO React Native New Architecture consumer

React Native **0.81.6** consumer. `android/gradle.properties` pins
`newArchEnabled=true`. It validates compile/link compatibility with current
RustO release artifacts while New Architecture is enabled.

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
xcodebuild -workspace ios/RustONewArchExample.xcworkspace -scheme RustONewArchExample \
  -configuration Debug -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 15' build
```

RustO remains bridge-module based; this app does **not** claim TurboModule or
codegen implementation. App does not invoke real OCR automatically. Models and
native runtime required for inference.
