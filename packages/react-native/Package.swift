// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "react-native-rusto",
  platforms: [
    .iOS(.v13),
    .macOS(.v11)
  ],
  products: [
    .library(
      name: "Rusto",
      targets: ["Rusto"]
    )
  ],
  dependencies: [],
  targets: [
    .target(
      name: "Rusto",
      dependencies: [],
      path: "ios",
      sources: [
        "Rusto.mm"
      ],
      publicHeadersPath: "."
    )
  ]
)
