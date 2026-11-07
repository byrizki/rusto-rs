export type OcrDemoLocale = 'en' | 'id';

export interface OcrDemoStrings {
  toolbar: {
    modelLabel: string;
    statusReady: string;
    statusLoading: string;
    statusError: string;
    retry: string;
    settings: string;
    runOcr: string;
    rerunOcr: string;
    processing: string;
    downloading: string;
    expandView: string;
    exitExpandedView: string;
  };
  samples: {
    label: string;
    customFile: string;
    uploadBtn: string;
    dropOrPasteHint: string;
    names: Record<string, string>;
  };
  canvas: {
    dropOverlayTitle: string;
    dropOverlaySub: string;
    boxesOn: string;
    boxesOff: string;
    resetZoom: string;
    replaceImage: string;
    copiedItem: string;
    hintIdle: string;
    hintActive: string;
  };
  results: {
    tabResults: string;
    tabSpatial: string;
    tabTable: string;
    tabJson: string;
    tabCode: string;
    modeLines: string;
    modeWords: string;
    searchPlaceholder: string;
    copy: string;
    copied: string;
    download: string;
    idleTitle: string;
    idleDescription: string;
    idleAction: string;
    idlePrivacyNote: string;
    emptyFilteredTitle: string;
    emptyNoMatchTitle: string;
    emptyNoMatchAction: string;
    tableViewGrid: string;
    tableViewRaw: string;
    tableHint: string;
    colText: string;
    colScore: string;
    colBox: string;
    footerNoRun: string;
    footerModified: string;
    footerViewCode: string;
  };
  settings: {
    title: string;
    resetDefaults: string;
    presetsLabel: string;
    tabPreDetection: string;
    tabDetection: string;
    tabRecognition: string;
    tabRuntime: string;
    close: string;
    applyAndRun: string;
    modifiedHint: string;
  };
}
