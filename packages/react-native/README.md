# react-native-rusto

React Native bindings for RustO! OCR library.

## Installation

```sh
npm install react-native-rusto
```

## Usage

```js
import { detectText } from "react-native-rusto";

// ...

const result = await detectText("path/to/image.jpg");
```

## Setup

This package relies on the native `RustO` libraries for iOS and Android. They are bundled with this package.
