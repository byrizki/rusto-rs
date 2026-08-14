# RustO React Native legacy-architecture consumer

React Native 0.70.15 source consumer. `android/gradle.properties` pins
`newArchEnabled=false`; `ios/Podfile` pins `RCT_NEW_ARCH_ENABLED=0`.

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
xcodebuild -workspace ios/RustOLegacyExample.xcworkspace -scheme RustOLegacyExample \
  -configuration Debug -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 15' build
```

This app does not invoke real OCR automatically; models/native runtime must be
supplied for runtime inference.
