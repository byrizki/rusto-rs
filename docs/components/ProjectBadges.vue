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
    alt: 'NuGet',
    src: 'https://img.shields.io/nuget/v/RustODotnet.svg?logo=nuget&logoColor=white&color=004880',
    href: 'https://www.nuget.org/packages/RustODotnet',
  },
  {
    alt: 'npm',
    src: 'https://img.shields.io/npm/v/react-native-rusto.svg?logo=npm&logoColor=white&color=CB3837',
    href: 'https://www.npmjs.com/package/react-native-rusto',
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
    alt: 'License',
    src: 'https://img.shields.io/badge/License-MIT-yellow.svg',
    href: 'https://opensource.org/licenses/MIT',
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
