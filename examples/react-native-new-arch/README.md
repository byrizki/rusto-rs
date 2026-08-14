# RustO React Native New Architecture consumer

React Native 0.70.15 source consumer. `android/gradle.properties` pins
`newArchEnabled=true`; `ios/Podfile` pins `RCT_NEW_ARCH_ENABLED=1`.

```bash
npm ci
npm test
npm run typecheck
(cd android && ./gradlew :app:assembleDebug)
```

Release CI first builds current AAR/XCFramework/NPM artifacts, then compiles this
consumer against those artifacts. For local iOS testing, place a built
`RustO.xcframework` in `../../packages/ios/`, then install pods and compile:

```bash
(cd ios && pod install)
xcodebuild -workspace ios/RustONewArchExample.xcworkspace -scheme RustONewArchExample \
  -configuration Debug -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 15' build
```

This gate proves compile/link compatibility while New Architecture is enabled. Current
library remains bridge-module based; no TurboModule/codegen spec is claimed. App
does not invoke OCR automatically; models/native runtime are runtime requirements.
