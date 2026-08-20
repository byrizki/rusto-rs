# RustO! Documentation Site

> Documentation site for [RustO!](https://github.com/byrizki/rusto-rs) — High-Performance, Pure Rust OCR Engine & Multi-Platform Toolkit based on RapidOCR with PaddleOCR engine and MNN inference.

Built with [Nuxt](https://nuxt.com), [Docus](https://docus.dev), and [Nuxt Content](https://content.nuxt.com/), deployed to [GitHub Pages](https://byrizki.github.io/rusto-rs/).

## 🚀 Quick Start

```bash
# Install dependencies
yarn install

# Start development server
yarn dev
```

The documentation site will run at `http://localhost:3000`.

## 📁 Project Structure

```
docs/
├── content/
│   ├── en/                     # English documentation
│   │   ├── index.md            # Landing page
│   │   ├── 01.getting-started/ # Getting started & installation
│   │   ├── 02.how-to/          # How-to guides per platform (Rust, .NET, RN, iOS, Android, C FFI, CLI)
│   │   ├── 03.advance-guide/   # Advanced topics & tuning (Models, Multilingual, Spatial, Preprocessing, Benchmarks)
│   │   └── 04.api-reference/   # API Reference (InitializeConfig, OcrRunOptions, TextResult)
│   └── id/                     # Indonesian documentation
├── components/                 # Vue components (AppHeaderLogo, ProjectBadges)
├── public/                     # Static assets (favicon, images)
├── nuxt.config.ts              # Nuxt configuration
├── app.config.ts               # Docus theme & branding configuration
└── package.json
```

## ⚡ Built With

- [Nuxt 4](https://nuxt.com) — Web framework
- [Docus](https://docus.dev) / [Nuxt Content](https://content.nuxt.com/) — Markdown & docs theme
- [Nuxt i18n](https://i18n.nuxt.com/) — Internationalization (English & Indonesian)

## 🚀 Deployment

Deploys automatically to GitHub Pages on every push to `main` that touches the `docs/**` path via the [deploy-docs workflow](../.github/workflows/deploy-docs.yml).

To build manually:

```bash
NUXT_APP_BASE_URL=/rusto-rs/ yarn build
```

Output will be generated in `.output/public/`.

## 📄 License

[MIT License](https://opensource.org/licenses/MIT)
