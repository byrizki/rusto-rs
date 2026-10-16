# Contributing to RustO! 🦀

Thank you for your interest in contributing to **RustO!**! We welcome community contributions of all kinds: bug reports, documentation improvements, benchmark datasets, model integrations, and new cross-platform language bindings.

RustO! is an independent, community-driven open-source project licensed under the [MIT License](LICENSE).

---

## Code of Conduct

All contributors and community participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat everyone with respect and kindness.

---

## Ways to Contribute

1. **Reporting Bugs**: Check the [GitHub Issues](https://github.com/byrizki/rusto-rs/issues) tracker first. If your issue is new, open a bug report detailing:
   - Operating system and architecture (e.g., Linux x86_64, macOS Apple Silicon, Android arm64)
   - Rust / language binding version
   - Model tier used (PP-OCRv6, PP-OCRv5, PP-OCRv4)
   - Minimal reproduction code and sample image (if not confidential)
2. **Suggesting Enhancements**: Propose feature ideas or API improvements in GitHub Discussions or Issues.
3. **Improving Documentation**: Fix typos, clarify explanations, or add guides in the `docs/` folder.
4. **Submitting Pull Requests (PRs)**: Implement fixes, optimizations, or new features.

---

## Development Setup

### Prerequisites

- **Rust**: Latest stable toolchain (`rustup update stable`)
- **Node.js & Yarn**: Node 18+ and Yarn 4 (for `docs/` and `@rustors/*` web packages)
- **Git LFS**: Required for test assets and sample images

### Building and Testing

```bash
# Clone the repository
git clone https://github.com/byrizki/rusto-rs.git
cd rusto-rs

# Build the core engine
cargo build

# Run core unit and integration tests
cargo test

# Run tests with OpenCV verification backend (optional, requires opencv-sys)
cargo test --features use-opencv

# Run code linter
cargo clippy --all-targets -- -D warnings

# Check code formatting
cargo fmt --check
```

### Documentation

The documentation website is built with Nuxt Content and Docus in the `docs/` directory:

```bash
cd docs
yarn install
yarn dev
```

---

## Pull Request Guidelines

1. **Branch Naming**: Use descriptive branch names like `feat/add-onnx-quantization`, `fix/dbnet-dilation`, or `docs/update-quickstart`.
2. **Keep Changes Focused**: Make surgical, single-purpose commits. Avoid combining unrelated refactors with bug fixes.
3. **Tests**: Add test coverage for new functionality or bug fixes where applicable.
4. **Format & Lint**: Ensure `cargo fmt --check` and `cargo clippy` pass cleanly before submitting.
5. **Signed Commits**: We encourage GPG or SSH signed commits.

---

## Security

For security vulnerabilities, please refer to our [Security Policy](SECURITY.md) and do not file public issues.

---

## License

By contributing to RustO!, you agree that your contributions will be licensed under the [MIT License](LICENSE).
