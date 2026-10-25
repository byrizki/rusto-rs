const { existsSync } = require('node:fs');
const { resolve } = require('node:path');

function loadNativeAddon() {
  const candidates = [
    resolve(__dirname, 'rusto.node'),
    resolve(__dirname, '../../target/debug/librusto_node.so'),
    resolve(__dirname, '../../target/release/librusto_node.so'),
    resolve(__dirname, '../../target/debug/rusto.node'),
    resolve(__dirname, '../../target/release/rusto.node'),
  ];

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      try {
        if (candidate.endsWith('.node')) {
          return require(candidate);
        }
        const mod = { exports: {} };
        process.dlopen(mod, candidate);
        return mod.exports;
      } catch (_) {}
    }
  }

  throw new Error('Failed to load @rustors/node native N-API addon.');
}

const native = loadNativeAddon();
const { RustO: RawRustO } = native;

/**
 * High-performance, pure-Rust OCR engine for Node.js via native N-API addons.
 */
class RustO {
  constructor(raw) {
    this._raw = raw;
  }

  /**
   * Initializes the OCR engine with configuration options or a JSON configuration string.
   *
   * @param {import('./index').InitializeConfig | string} config - Initialization options or JSON string.
   * @returns {RustO} Initialized engine instance.
   */
  static initialize(config) {
    const json = typeof config === 'string' ? config : JSON.stringify(config);
    const raw = RawRustO.initialize(json);
    return new RustO(raw);
  }

  /**
   * Detects text in an image file at the given path.
   *
   * @param {string} path - Image file path.
   * @param {import('./index').OcrRunOptions | string} [options] - Runtime options or JSON string.
   * @returns {import('./index').TextResult[]} Structured OCR results.
   */
  detectTextFile(path, options) {
    const opts = typeof options === 'string' ? options : (options ? JSON.stringify(options) : null);
    const rawResult = this._raw.detectTextFile(path, opts);
    try {
      return JSON.parse(rawResult);
    } catch {
      return rawResult;
    }
  }

  /**
   * Detects text from an in-memory image buffer.
   *
   * @param {Buffer | Uint8Array} data - Image file buffer.
   * @param {import('./index').OcrRunOptions | string} [options] - Runtime options or JSON string.
   * @returns {import('./index').TextResult[]} Structured OCR results.
   */
  detectTextData(data, options) {
    const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data);
    const opts = typeof options === 'string' ? options : (options ? JSON.stringify(options) : null);
    const rawResult = this._raw.detectTextData(buffer, opts);
    try {
      return JSON.parse(rawResult);
    } catch {
      return rawResult;
    }
  }

  /**
   * Detects text in an image file and returns a reconstructed 2D spatial layout.
   *
   * @param {string} path - Image file path.
   * @param {import('./index').OcrRunOptions | string} [options] - Runtime options or JSON string.
   * @returns {string} Reconstructed spatial text layout.
   */
  detectTextSpatial(path, options) {
    const opts = typeof options === 'string' ? options : (options ? JSON.stringify(options) : null);
    return this._raw.detectTextSpatial(path, opts);
  }

  /**
   * Returns the underlying RustO engine version.
   *
   * @returns {string} Version string.
   */
  version() {
    return this._raw.version();
  }
}

module.exports = { RustO };
