export interface ModelAssetFiles {
  detection: string;
  recognition: string;
  dictionary: string;
}

export interface ReleaseAssetInfo {
  name: string;
  size: number;
  browser_download_url: string;
}

export const DEFAULT_MODELS_TAG = 'models-rten-v0.4.0';
export const GITHUB_RELEASE_BASE = 'https://github.com/byrizki/rusto-rs/releases/download';
export const GITHUB_API_BASE = 'https://api.github.com/repos/byrizki/rusto-rs';
export const JSDELIVR_CDN_BASE = 'https://cdn.jsdelivr.net/npm';

export const MODEL_RELEASE_ASSETS: Record<string, ModelAssetFiles> = {
  indraocr: {
    detection: 'indra_det.rten',
    recognition: 'indra_rec.rten',
    dictionary: 'indra_meta.json',
  },
  'ppocrv6-tiny': {
    detection: 'ppocrv6_tiny_det.rten',
    recognition: 'ppocrv6_tiny_rec.rten',
    dictionary: 'ppocrv6_dict.txt',
  },
  'ppocrv6-tiny-int8': {
    detection: 'ppocrv6_tiny_int8_det.rten',
    recognition: 'ppocrv6_tiny_int8_rec.rten',
    dictionary: 'ppocrv6_dict.txt',
  },
  'ppocrv6-small': {
    detection: 'ppocrv6_small_det.rten',
    recognition: 'ppocrv6_small_rec.rten',
    dictionary: 'ppocrv6_dict.txt',
  },
  'ppocrv6-small-int8': {
    detection: 'ppocrv6_small_int8_det.rten',
    recognition: 'ppocrv6_small_int8_rec.rten',
    dictionary: 'ppocrv6_dict.txt',
  },
  'ppocrv6-medium': {
    detection: 'ppocrv6_medium_det.rten',
    recognition: 'ppocrv6_medium_rec.rten',
    dictionary: 'ppocrv6_dict.txt',
  },
  'ppocrv6-medium-int8': {
    detection: 'ppocrv6_medium_int8_det.rten',
    recognition: 'ppocrv6_medium_int8_rec.rten',
    dictionary: 'ppocrv6_dict.txt',
  },
  'ppocrv5-mobile': {
    detection: 'ppocrv5_mobile_det.rten',
    recognition: 'ppocrv5_mobile_rec.rten',
    dictionary: 'ppocrv5_dict.txt',
  },
  'ppocrv5-server': {
    detection: 'ppocrv5_server_det.rten',
    recognition: 'ppocrv5_server_rec.rten',
    dictionary: 'ppocrv5_dict.txt',
  },
  'ppocrv4-mobile': {
    detection: 'ppocrv4_mobile_det.rten',
    recognition: 'ppocrv4_mobile_rec.rten',
    dictionary: 'ppocrv4_dict.txt',
  },
  'ppocrv4-server': {
    detection: 'ppocrv4_server_det.rten',
    recognition: 'ppocrv4_server_rec.rten',
    dictionary: 'ppocrv4_dict.txt',
  },
};

export function getModelCandidateUrls(
  modelId: string,
  baseURL: string = '/',
  tag: string = DEFAULT_MODELS_TAG
): {
  detection: string[];
  recognition: string[];
  dictionary: string[];
} {
  const asset = MODEL_RELEASE_ASSETS[modelId] || MODEL_RELEASE_ASSETS['indraocr']!;
  const cleanBase = baseURL.endsWith('/') ? baseURL.slice(0, -1) : baseURL;
  const isIndra = modelId === 'indraocr' || modelId === 'indra';
  const dictFile = isIndra ? 'meta.json' : 'dict.txt';

  return {
    detection: [
      `${cleanBase}/models/${modelId}/det.rten`,
      `${cleanBase}/models/${asset.detection}`,
      `${GITHUB_RELEASE_BASE}/${tag}/${asset.detection}`,
    ],
    recognition: [
      `${cleanBase}/models/${modelId}/rec.rten`,
      `${cleanBase}/models/${asset.recognition}`,
      `${GITHUB_RELEASE_BASE}/${tag}/${asset.recognition}`,
    ],
    dictionary: [
      `${cleanBase}/models/${modelId}/${dictFile}`,
      `${cleanBase}/models/${asset.dictionary}`,
      `${GITHUB_RELEASE_BASE}/${tag}/${asset.dictionary}`,
    ],
  };
}

export function getModelCdnUrls(
  modelId: string,
  tag: string = DEFAULT_MODELS_TAG
): {
  detection: string;
  recognition: string;
  dictionary: string;
} {
  const asset = MODEL_RELEASE_ASSETS[modelId] || MODEL_RELEASE_ASSETS['indraocr']!;
  return {
    detection: `${GITHUB_RELEASE_BASE}/${tag}/${asset.detection}`,
    recognition: `${GITHUB_RELEASE_BASE}/${tag}/${asset.recognition}`,
    dictionary: `${GITHUB_RELEASE_BASE}/${tag}/${asset.dictionary}`,
  };
}

export async function fetchGitHubReleaseModelIndex(
  tag: string = DEFAULT_MODELS_TAG
): Promise<ReleaseAssetInfo[]> {
  try {
    const res = await fetch(`${GITHUB_API_BASE}/releases/tags/${tag}`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data.assets) && data.assets.length > 0) {
        return data.assets.map(
          (a: { name: string; size: number; browser_download_url: string }) => ({
            name: a.name,
            size: a.size,
            browser_download_url: a.browser_download_url,
          })
        );
      }
    }

    // Fallback: if tag is unreleased (404), query recent releases for published models
    const releasesRes = await fetch(`${GITHUB_API_BASE}/releases?per_page=10`);
    if (releasesRes.ok) {
      const releases = await releasesRes.json();
      const modelRelease = releases.find(
        (r: { tag_name: string; assets: unknown[] }) =>
          r.tag_name?.startsWith('models-rten-v') && Array.isArray(r.assets) && r.assets.length > 0
      );
      if (modelRelease && Array.isArray(modelRelease.assets)) {
        return modelRelease.assets.map(
          (a: { name: string; size: number; browser_download_url: string }) => ({
            name: a.name,
            size: a.size,
            browser_download_url: a.browser_download_url,
          })
        );
      }
    }
  } catch {
    // Non-blocking fallback
  }
  return [];
}
