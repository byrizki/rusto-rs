// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "RustO",
  platforms: [
    .iOS(.v12),
    .macOS(.v11)
  ],
  products: [
    .library(
      name: "RustO",
      targets: ["RustO"]
    )
  ],
  dependencies: [],
  targets: [
    .target(
      name: "RustOFFI",
      path: "src/generated",
      publicHeadersPath: "."
    ),
    .target(
      name: "RustO",
      dependencies: ["RustOFFI"],
      path: "src",
      exclude: ["generated"],
      sources: [
        "RustO.swift"
      ],
      publicHeadersPath: "."
    )
  ]
)
