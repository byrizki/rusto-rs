<script setup lang="ts">
import type { Collections, DocsCollectionItem } from '@nuxt/content';

definePageMeta({
  layout: 'docs',
});

const route = useRoute();
const { locale, isEnabled } = useDocusI18n();
const collectionName = computed(() => (isEnabled.value ? `docs_${locale.value}` : 'docs'));

const { data: page } = await useAsyncData(
  `page-demo-${locale.value}`,
  () =>
    queryCollection(collectionName.value as keyof Collections)
      .path(route.path)
      .first() as Promise<DocsCollectionItem>
);

const title = computed(
  () =>
    page.value?.title || (locale.value === 'id' ? 'Demo Web Interaktif' : 'Interactive Web Demo')
);

const description = computed(
  () =>
    page.value?.description ||
    (locale.value === 'id'
      ? 'Coba OCR RustO! langsung di browser Anda bertenaga WebAssembly dan model PaddleOCR.'
      : 'Experience RustO! OCR directly in your browser powered by WebAssembly and pre-trained PaddleOCR models.')
);

useSeo({
  title: title.value,
  description: description.value,
  type: 'article',
});
</script>

<template>
  <UPage>
    <UPageHeader :title="title" :description="description" />
    <UPageBody>
      <ContentRenderer v-if="page" :value="page" />
      <WebOcrDemo v-else />
    </UPageBody>
  </UPage>
</template>
