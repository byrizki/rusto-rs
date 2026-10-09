export default defineAppConfig({
  header: {
    logo: {
      favicon: '/favicon.svg',
    },
  },
  docus: {
    title: 'RustO!',
    description:
      'High-Performance, Pure Rust OCR Engine & Multi-Platform Toolkit powered by RapidOCR, PaddleOCR models, and pure-Rust RTen inference.',
    socials: {
      github: 'byrizki/rusto-rs',
    },
  },
  ui: {
    colors: {
      primary: 'orange',
      neutral: 'zinc',
    },
    contentSurround: {
      slots: {
        root: 'grid grid-cols-1 sm:grid-cols-2 gap-4',
        link: 'group block p-4 rounded-lg border border-default hover:bg-elevated/50 outline-primary/25 focus-visible:outline-3 focus-visible:border-primary transition-colors',
        linkLeading:
          'inline-flex items-center rounded-full p-1.5 bg-elevated group-hover:bg-primary/10 ring ring-accented mb-2.5 group-hover:ring-primary/50 transition',
        linkTitle: 'font-medium text-[15px] text-highlighted mb-1 truncate',
        linkDescription: 'text-sm text-muted line-clamp-2',
      },
      variants: {
        direction: {
          right: {
            link: 'text-end',
          },
        },
      },
    },
  },
});
