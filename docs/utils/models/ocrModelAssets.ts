export interface ModelAssetFiles {
  detection: string;
  recognition: string;
  dictionary: string;
  legacyDetection?: string;
  legacyRecognition?: string;
  legacyDictionary?: string;
}

export function makeModelAsset(
  folder: string,
  det: string,
  rec: string,
  dict: string,
  legacyDet?: string,
  legacyRec?: string,
  legacyDict?: string
): ModelAssetFiles {
  return {
    detection: `${folder}/${det}`,
    recognition: `${folder}/${rec}`,
    dictionary: `${folder}/${dict}`,
    legacyDetection: legacyDet,
    legacyRecognition: legacyRec,
    legacyDictionary: legacyDict,
  };
}

export const MODEL_RELEASE_ASSETS: Record<string, ModelAssetFiles> = {
  indraocr: makeModelAsset(
    'indraocr',
    'det.mnn',
    'rec.mnn',
    'meta.json',
    'indra_det.mnn',
    'indra_rec.mnn',
    'indra_meta.json'
  ),
  'indraocr-cjk': makeModelAsset(
    'indraocr',
    'det.mnn',
    'rec_cjk.onnx',
    'meta.json',
    'indra_det.mnn',
    'indra_rec_cjk.onnx',
    'indra_meta.json'
  ),
  'indraocr-world': makeModelAsset(
    'indraocr',
    'det.mnn',
    'rec_world.onnx',
    'meta.json',
    'indra_det.mnn',
    'indra_rec_world.onnx',
    'indra_meta.json'
  ),

  'ppocrv6-tiny': makeModelAsset(
    'ppocrv6-tiny',
    'det.mnn',
    'rec.mnn',
    'dict.txt',
    'ppocrv6_tiny_det.mnn',
    'ppocrv6_tiny_rec.mnn',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-tiny-int8': makeModelAsset(
    'ppocrv6-tiny-int8',
    'det.onnx',
    'rec.onnx',
    'dict.txt',
    'ppocrv6_tiny_int8_det.onnx',
    'ppocrv6_tiny_int8_rec.onnx',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-small': makeModelAsset(
    'ppocrv6-small',
    'det.onnx',
    'rec.onnx',
    'dict.txt',
    'ppocrv6_small_det.onnx',
    'ppocrv6_small_rec.onnx',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-small-int8': makeModelAsset(
    'ppocrv6-small-int8',
    'det.onnx',
    'rec.onnx',
    'dict.txt',
    'ppocrv6_small_int8_det.onnx',
    'ppocrv6_small_int8_rec.onnx',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-medium': makeModelAsset(
    'ppocrv6-medium',
    'det.onnx',
    'rec.onnx',
    'dict.txt',
    'ppocrv6_medium_det.onnx',
    'ppocrv6_medium_rec.onnx',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-medium-int8': makeModelAsset(
    'ppocrv6-medium-int8',
    'det.onnx',
    'rec.onnx',
    'dict.txt',
    'ppocrv6_medium_int8_det.onnx',
    'ppocrv6_medium_int8_rec.onnx',
    'ppocrv6_dict.txt'
  ),

  'ppocrv5-mobile': makeModelAsset(
    'ppocrv5-mobile',
    'det.onnx',
    'rec.onnx',
    'dict.txt',
    'ppocrv5_mobile_det.onnx',
    'ppocrv5_mobile_rec.onnx',
    'ppocrv5_dict.txt'
  ),
  'ppocrv4-mobile': makeModelAsset(
    'ppocrv4-mobile',
    'det.onnx',
    'rec.onnx',
    'dict.txt',
    'ppocrv4_mobile_det.onnx',
    'ppocrv4_mobile_rec.onnx',
    'ppocrv4_dict.txt'
  ),

  'ppocrv4-rec-japan': makeModelAsset(
    'multilingual',
    'ppocrv4-mobile/det.onnx',
    'ppocrv4_rec_japan.onnx',
    'ppocrv4_dict_japan.txt'
  ),
  'ppocrv4-rec-chinese-cht': makeModelAsset(
    'multilingual',
    'ppocrv4-mobile/det.onnx',
    'ppocrv4_rec_chinese_cht.onnx',
    'ppocrv4_dict_chinese_cht.txt'
  ),
  'ppocrv5-rec-korean': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.onnx',
    'ppocrv5_rec_korean.onnx',
    'ppocrv5_dict_korean.txt'
  ),
  'ppocrv5-rec-arabic': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.onnx',
    'ppocrv5_rec_arabic.onnx',
    'ppocrv5_dict_arabic.txt'
  ),
  'ppocrv5-rec-cyrillic': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.onnx',
    'ppocrv5_rec_cyrillic.rten',
    'ppocrv5_dict_cyrillic.txt'
  ),
  'ppocrv5-rec-devanagari': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.onnx',
    'ppocrv5_rec_devanagari.onnx',
    'ppocrv5_dict_devanagari.txt'
  ),
  'ppocrv5-rec-latin': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.onnx',
    'ppocrv5_rec_latin.onnx',
    'ppocrv5_dict_latin.txt'
  ),
  'ppocrv5-rec-th': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.onnx',
    'ppocrv5_rec_th.rten',
    'ppocrv5_dict_th.txt'
  ),
};
