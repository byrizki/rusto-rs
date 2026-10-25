# RustO for Python

[![PyPI](https://img.shields.io/pypi/v/rusto.svg?logo=pypi&logoColor=white&color=blue)](https://pypi.org/project/rusto/)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Rust OCR library for Python based on RapidOCR and PaddleOCR architectures.

> 📖 **Full Documentation & Guides**: [byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)

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

## 📖 Documentation

For full documentation, guides, and API reference:
- [Python Guide](https://byrizki.com/rusto-rs/how-to/use-python)
- [Full Documentation](https://byrizki.com/rusto-rs/)

## License

MIT

