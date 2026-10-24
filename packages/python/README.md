# RustO for Python

[![PyPI](https://img.shields.io/pypi/v/rusto.svg?logo=pypi&logoColor=white&color=blue)](https://pypi.org/project/rusto/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Rust OCR library for Python based on RapidOCR and PaddleOCR architectures.

## Installation

```bash
pip install rusto
```

## Quick Start

```python
from rusto import RustO, InitializeConfig

# Initialize engine with preset or model paths
engine = RustO.initialize(InitializeConfig(template="ppv6"))

# Run OCR detection
result = engine.detect_text(path="document.png")
for line in result.results:
    print(f"{line.text} (score: {line.score:.2f})")
```

## License

MIT
