import { computed } from 'vue';
import { useRoute } from '#app';
import type { OcrDemoLocale, OcrDemoStrings } from '../types/ocrDemoI18n';

const enStrings: OcrDemoStrings = {
  toolbar: {
    modelLabel: 'Model',
    statusReady: 'Ready',
    statusLoading: 'Loading',
    statusError: 'Load failed',
    retry: 'Retry',
    settings: 'Settings',
    runOcr: 'Run OCR',
    rerunOcr: 'Re-run OCR',
    processing: 'Scanning...',
    downloading: 'Loading model...',
    expandView: 'Expand workspace',
    exitExpandedView: 'Exit expanded view',
  },
  samples: {
    label: 'Document',
    customFile: 'Custom upload',
    uploadBtn: 'Upload image',
    dropOrPasteHint: 'or drop / paste (Ctrl+V)',
    names: {
      invoice: 'Invoice',
      idcard: 'Passport',
      handwritten: 'Handwritten',
      'dense-newspaper': 'Newspaper',
    },
  },
  canvas: {
    dropOverlayTitle: 'Drop image to load',
    dropOverlaySub: 'PNG, JPG, WebP — processed locally in your browser',
    boxesOn: 'Boxes',
    boxesOff: 'Boxes hidden',
    resetZoom: 'Reset zoom',
    replaceImage: 'Upload image',
    copiedItem: 'Copied text',
    hintIdle: 'Drop or paste any image to test, then click Run OCR.',
    hintActive: 'Hover boxes to inspect • Click any box to copy text',
  },
  results: {
    tabResults: 'Text',
    tabSpatial: 'Spatial',
    tabTable: 'Table',
    tabJson: 'JSON',
    tabCode: 'Code',
    modeLines: 'Lines',
    modeWords: 'Words',
    searchPlaceholder: 'Filter detected text...',
    copy: 'Copy',
    copied: 'Copied',
    download: 'Export',
    idleTitle: 'Ready to scan locally',
    idleDescription:
      'Model weights download once and execute directly in your browser via WebAssembly.',
    idleAction: 'Run OCR on Document',
    idlePrivacyNote: '100% client-side • Zero server uploads',
    emptyFilteredTitle: 'No items match your search filter.',
    emptyNoMatchTitle: 'No text met the minimum confidence threshold.',
    emptyNoMatchAction: 'Adjust threshold in Settings',
    tableViewGrid: 'Grid',
    tableViewRaw: 'Raw TSV',
    tableHint: 'Tab-separated for Sheets & Excel',
    colText: 'Text',
    colScore: 'Score',
    colBox: 'Bounds (x, y, w, h)',
    footerNoRun: 'Not scanned yet',
    footerModified: 'Settings changed',
    footerViewCode: 'View SDK code',
  },
  settings: {
    title: 'OCR Pipeline Settings',
    resetDefaults: 'Reset defaults',
    presetsLabel: 'Preset',
    tabDetection: 'Detection',
    tabRecognition: 'Recognition',
    tabRuntime: 'Runtime',
    close: 'Done',
    applyAndRun: 'Apply & Run OCR',
    modifiedHint: 'Settings changed since last scan',
  },
};

const idStrings: OcrDemoStrings = {
  toolbar: {
    modelLabel: 'Model',
    statusReady: 'Siap',
    statusLoading: 'Memuat',
    statusError: 'Gagal memuat',
    retry: 'Coba lagi',
    settings: 'Pengaturan',
    runOcr: 'Jalankan OCR',
    rerunOcr: 'Jalankan Ulang',
    processing: 'Memindai...',
    downloading: 'Memuat model...',
    expandView: 'Perluas tampilan',
    exitExpandedView: 'Kecilkan tampilan',
  },
  samples: {
    label: 'Dokumen',
    customFile: 'Gambar kustom',
    uploadBtn: 'Unggah gambar',
    dropOrPasteHint: 'atau tarik / tempel (Ctrl+V)',
    names: {
      invoice: 'Faktur',
      idcard: 'Paspor',
      handwritten: 'Tulisan Tangan',
      'dense-newspaper': 'Koran Padat',
    },
  },
  canvas: {
    dropOverlayTitle: 'Lepaskan gambar di sini',
    dropOverlaySub: 'PNG, JPG, WebP — diproses langsung di peramban Anda',
    boxesOn: 'Kotak',
    boxesOff: 'Kotak disembunyikan',
    resetZoom: 'Atur ulang zoom',
    replaceImage: 'Unggah gambar',
    copiedItem: 'Teks disalin',
    hintIdle: 'Tarik atau tempel gambar apa pun, lalu klik Jalankan OCR.',
    hintActive: 'Arahkan kursor ke kotak untuk inspeksi • Klik kotak untuk menyalin',
  },
  results: {
    tabResults: 'Teks',
    tabSpatial: 'Spasial',
    tabTable: 'Tabel',
    tabJson: 'JSON',
    tabCode: 'Kode',
    modeLines: 'Baris',
    modeWords: 'Kata',
    searchPlaceholder: 'Cari teks terdeteksi...',
    copy: 'Salin',
    copied: 'Tersalin',
    download: 'Ekspor',
    idleTitle: 'Siap memindai secara lokal',
    idleDescription:
      'Bobot model diunduh sekali dan dijalankan langsung di peramban menggunakan WebAssembly.',
    idleAction: 'Jalankan OCR Sekarang',
    idlePrivacyNote: '100% sisi klien • Tanpa unggah ke server',
    emptyFilteredTitle: 'Tidak ada teks yang cocok dengan pencarian.',
    emptyNoMatchTitle: 'Tidak ada teks yang memenuhi batas skor minimum.',
    emptyNoMatchAction: 'Ubah batas skor di Pengaturan',
    tableViewGrid: 'Tabel',
    tableViewRaw: 'TSV Mentah',
    tableHint: 'Format tab siap tempel ke Sheets & Excel',
    colText: 'Teks',
    colScore: 'Skor',
    colBox: 'Kotak (x, y, l, t)',
    footerNoRun: 'Belum dipindai',
    footerModified: 'Pengaturan berubah',
    footerViewCode: 'Lihat kode SDK',
  },
  settings: {
    title: 'Pengaturan Pipeline OCR',
    resetDefaults: 'Atur ulang bawaan',
    presetsLabel: 'Preset',
    tabDetection: 'Deteksi',
    tabRecognition: 'Pengenalan',
    tabRuntime: 'Runtime',
    close: 'Selesai',
    applyAndRun: 'Terapkan & Jalankan',
    modifiedHint: 'Pengaturan berubah sejak pemindaian terakhir',
  },
};

export function useOcrDemoI18n() {
  const route = useRoute();

  const locale = computed<OcrDemoLocale>(() => {
    const path = route?.path || '';
    return path === '/id' || path.startsWith('/id/') ? 'id' : 'en';
  });

  const strings = computed<OcrDemoStrings>(() => (locale.value === 'id' ? idStrings : enStrings));

  return {
    locale,
    strings,
  };
}
