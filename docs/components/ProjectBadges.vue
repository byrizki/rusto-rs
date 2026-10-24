<script setup lang="ts">
const badges = [
  {
    alt: 'Crates.io',
    src: 'https://img.shields.io/crates/v/rusto-rs.svg?logo=rust&logoColor=white&color=orange',
    href: 'https://crates.io/crates/rusto-rs',
  },
  {
    alt: 'docs.rs',
    src: 'https://img.shields.io/docsrs/rusto-rs?logo=docs.rs&logoColor=white',
    href: 'https://docs.rs/rusto-rs',
  },
  {
    alt: 'PyPI',
    src: 'https://img.shields.io/pypi/v/rusto.svg?logo=pypi&logoColor=white&color=blue',
    href: 'https://pypi.org/project/rusto/',
  },
  {
    alt: 'npm (node)',
    src: 'https://img.shields.io/npm/v/@rustors/node.svg?logo=node.js&logoColor=white&color=339933&label=npm%20node',
    href: 'https://www.npmjs.com/package/@rustors/node',
  },
  {
    alt: 'pub.dev (dart)',
    src: 'https://img.shields.io/pub/v/rusto.svg?logo=dart&logoColor=white&color=0175C2&label=pub%20dart',
    href: 'https://pub.dev/packages/rusto',
  },
  {
    alt: 'pub.dev (flutter)',
    src: 'https://img.shields.io/pub/v/rusto_flutter.svg?logo=flutter&logoColor=white&color=02569B&label=pub%20flutter',
    href: 'https://pub.dev/packages/rusto_flutter',
  },
  {
    alt: 'NuGet',
    src: 'https://img.shields.io/nuget/v/RustODotnet.svg?logo=nuget&logoColor=white&color=004880',
    href: 'https://www.nuget.org/packages/RustODotnet',
  },
  {
    alt: 'npm (react-native)',
    src: 'https://img.shields.io/npm/v/@rustors/react-native.svg?logo=react&logoColor=white&color=61DAFB&label=npm%20rn',
    href: 'https://www.npmjs.com/package/@rustors/react-native',
  },
  {
    alt: 'CocoaPods',
    src: 'https://img.shields.io/cocoapods/v/RustO.svg?logo=cocoapods&logoColor=white&color=EE3322',
    href: 'https://cocoapods.org/pods/RustO',
  },
  {
    alt: 'JitPack',
    src: 'https://jitpack.io/v/byrizki/rusto-rs.svg',
    href: 'https://jitpack.io/#byrizki/rusto-rs',
  },
  {
    alt: 'Library CI',
    src: 'https://github.com/byrizki/rusto-rs/actions/workflows/ci.yml/badge.svg',
    href: 'https://github.com/byrizki/rusto-rs/actions/workflows/ci.yml',
  },
  {
    alt: 'Docs CI',
    src: 'https://github.com/byrizki/rusto-rs/actions/workflows/docs-ci.yml/badge.svg',
    href: 'https://github.com/byrizki/rusto-rs/actions/workflows/docs-ci.yml',
  },
  {
    alt: 'Build & Release',
    src: 'https://github.com/byrizki/rusto-rs/actions/workflows/build.yml/badge.svg',
    href: 'https://github.com/byrizki/rusto-rs/actions/workflows/build.yml',
  },
  {
    alt: 'License',
    src: 'https://img.shields.io/badge/License-MIT-yellow.svg',
    href: 'https://opensource.org/licenses/MIT',
  },
  {
    alt: 'Known Vulnerabilities',
    src: 'https://snyk.io/test/github/byrizki/rusto-rs/badge.svg',
    href: 'https://snyk.io/test/github/byrizki/rusto-rs',
  },
];
</script>

<template>
  <div class="flex flex-wrap items-center gap-3 my-4">
    <a
      v-for="badge in badges"
      :key="badge.alt"
      :href="badge.href"
      target="_blank"
      rel="noopener noreferrer"
      class="hover:opacity-80 transition-opacity inline-flex items-center"
    >
      <img :src="badge.src" :alt="badge.alt" class="h-[24px] w-auto" />
    </a>
  </div>
</template>
