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
    'det.rten',
    'rec.rten',
    'meta.json',
    'indra_det.rten',
    'indra_rec.rten',
    'indra_meta.json'
  ),
  'indraocr-cjk': makeModelAsset(
    'indraocr',
    'det.rten',
    'rec_cjk.rten',
    'meta.json',
    'indra_det.rten',
    'indra_rec_cjk.rten',
    'indra_meta.json'
  ),
  'indraocr-world': makeModelAsset(
    'indraocr',
    'det.rten',
    'rec_world.rten',
    'meta.json',
    'indra_det.rten',
    'indra_rec_world.rten',
    'indra_meta.json'
  ),

  'ppocrv6-tiny': makeModelAsset(
    'ppocrv6-tiny',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv6_tiny_det.rten',
    'ppocrv6_tiny_rec.rten',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-tiny-int8': makeModelAsset(
    'ppocrv6-tiny-int8',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv6_tiny_int8_det.rten',
    'ppocrv6_tiny_int8_rec.rten',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-small': makeModelAsset(
    'ppocrv6-small',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv6_small_det.rten',
    'ppocrv6_small_rec.rten',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-small-int8': makeModelAsset(
    'ppocrv6-small-int8',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv6_small_int8_det.rten',
    'ppocrv6_small_int8_rec.rten',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-medium': makeModelAsset(
    'ppocrv6-medium',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv6_medium_det.rten',
    'ppocrv6_medium_rec.rten',
    'ppocrv6_dict.txt'
  ),
  'ppocrv6-medium-int8': makeModelAsset(
    'ppocrv6-medium-int8',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv6_medium_int8_det.rten',
    'ppocrv6_medium_int8_rec.rten',
    'ppocrv6_dict.txt'
  ),

  'ppocrv5-mobile': makeModelAsset(
    'ppocrv5-mobile',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv5_mobile_det.rten',
    'ppocrv5_mobile_rec.rten',
    'ppocrv5_dict.txt'
  ),
  'ppocrv5-server': makeModelAsset(
    'ppocrv5-server',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv5_server_det.rten',
    'ppocrv5_server_rec.rten',
    'ppocrv5_dict.txt'
  ),

  'ppocrv4-mobile': makeModelAsset(
    'ppocrv4-mobile',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv4_mobile_det.rten',
    'ppocrv4_mobile_rec.rten',
    'ppocrv4_dict.txt'
  ),
  'ppocrv4-server': makeModelAsset(
    'ppocrv4-server',
    'det.rten',
    'rec.rten',
    'dict.txt',
    'ppocrv4_server_det.rten',
    'ppocrv4_server_rec.rten',
    'ppocrv4_dict.txt'
  ),

  'ppocrv4-rec-japan': makeModelAsset(
    'multilingual',
    'ppocrv4-mobile/det.rten',
    'ppocrv4_rec_japan.rten',
    'ppocrv4_dict_japan.txt'
  ),
  'ppocrv4-rec-chinese-cht': makeModelAsset(
    'multilingual',
    'ppocrv4-mobile/det.rten',
    'ppocrv4_rec_chinese_cht.rten',
    'ppocrv4_dict_chinese_cht.txt'
  ),
  'ppocrv5-rec-korean': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.rten',
    'ppocrv5_rec_korean.rten',
    'ppocrv5_dict_korean.txt'
  ),
  'ppocrv5-rec-arabic': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.rten',
    'ppocrv5_rec_arabic.rten',
    'ppocrv5_dict_arabic.txt'
  ),
  'ppocrv5-rec-cyrillic': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.rten',
    'ppocrv5_rec_cyrillic.rten',
    'ppocrv5_dict_cyrillic.txt'
  ),
  'ppocrv5-rec-devanagari': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.rten',
    'ppocrv5_rec_devanagari.rten',
    'ppocrv5_dict_devanagari.txt'
  ),
  'ppocrv5-rec-latin': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.rten',
    'ppocrv5_rec_latin.rten',
    'ppocrv5_dict_latin.txt'
  ),
  'ppocrv5-rec-th': makeModelAsset(
    'multilingual',
    'ppocrv5-mobile/det.rten',
    'ppocrv5_rec_th.rten',
    'ppocrv5_dict_th.txt'
  ),
};
