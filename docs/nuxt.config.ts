import { fileURLToPath } from 'node:url';

const baseURL = process.env.NUXT_APP_BASE_URL || '/rusto-rs/';

export default defineNuxtConfig({
  alias: {
    '@rustors/web': fileURLToPath(new URL('../packages/web/src/index.ts', import.meta.url)),
    '@rustors/model-ppocrv6-tiny': fileURLToPath(
      new URL('../packages/models/ppocrv6-tiny/src/index.ts', import.meta.url)
    ),
  },
  modules: ['@nuxtjs/i18n'],
  app: {
    head: {
      title: 'RustO! - High-Performance Pure Rust OCR Engine & Multi-Platform Toolkit',
      meta: [
        {
          name: 'description',
          content:
            'High-Performance Optical Character Recognition (OCR) engine and multi-platform toolkit written in pure Rust.',
        },
      ],
      link: [{ rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    },
    baseURL,
  },
  i18n: {
    defaultLocale: 'en',
    locales: [
      {
        code: 'en',
        name: 'English',
      },
      {
        code: 'id',
        name: 'Indonesia',
      },
    ],
  },
  content: {
    build: {
      markdown: {
        highlight: {
          theme: {
            default: 'github-light',
            dark: 'github-dark',
            sepia: 'monokai',
          },
          langs: [
            'js',
            'ts',
            'jsx',
            'tsx',
            'json',
            'bash',
            'sh',
            'toml',
            'csharp',
            'rust',
            'html',
            'swift',
            'kotlin',
            'groovy',
            'ruby',
            'c',
            'cpp',
            'mermaid',
          ],
        },
      },
    },
  },
});
