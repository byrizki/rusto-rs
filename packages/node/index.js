const { existsSync } = require('node:fs');
const { resolve } = require('node:path');

function loadNativeAddon() {
  const candidates = [
    resolve(__dirname, 'rusto.node'),
    resolve(__dirname, '../../target/release/librusto_node.so'),
    resolve(__dirname, '../../target/debug/librusto_node.so'),
    resolve(__dirname, '../../target/release/rusto.node'),
    resolve(__dirname, '../../target/debug/rusto.node'),
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
const INTERNAL_RAW_TOKEN = Symbol('rusto.raw');

/**
 * Fluent builder for chaining text detection options and running inference.
 */
class DetectTextBuilder {
  constructor(engine, source = null) {
    this._engine = engine;
    this._source = source;
    this._rgb = null;
    this._options = {};
  }

  image(source) {
    this._source = source;
    this._rgb = null;
    return this;
  }

  rgb(pixels, width, height, stride) {
    this._rgb = { pixels, width, height, stride };
    this._source = null;
    return this;
  }

  withAutoUnwarp(enabled = true) {
    this._options.autoUnwarp = enabled;
    return this;
  }

  withLayoutReading(enabled = true) {
    this._options.layoutReading = enabled;
    return this;
  }

  withOrientationDetection(enabled = true) {
    this._options.orientationDetection = enabled;
    return this;
  }

  withLineClassification(enabled = true) {
    this._options.lineClassification = enabled;
    return this;
  }

  withPreDetection(options) {
    this._options.preDetection = Object.assign({}, this._options.preDetection, options);
    return this;
  }

  withDetection(options) {
    this._options.detection = Object.assign({}, this._options.detection, options);
    return this;
  }

  withPostDetection(options) {
    this._options.postDetection = Object.assign({}, this._options.postDetection, options);
    return this;
  }

  withPreRecognition(options) {
    this._options.preRecognition = Object.assign({}, this._options.preRecognition, options);
    return this;
  }

  withRecognition(options) {
    this._options.recognition = Object.assign({}, this._options.recognition, options);
    return this;
  }

  withPostRecognition(options) {
    this._options.postRecognition = Object.assign({}, this._options.postRecognition, options);
    return this;
  }

  asLines() {
    this._options.output = 'lines';
    return this;
  }

  asWords() {
    this._options.output = 'words';
    return this;
  }

  asSpatial() {
    this._options.output = 'spatial';
    return this;
  }

  asTsv() {
    this._options.output = 'tsv';
    return this;
  }

  withOutput(granularity) {
    this._options.output = granularity;
    return this;
  }

  withOptions(options) {
    this._options = Object.assign({}, this._options, options);
    return this;
  }

  run(source) {
    if (source !== undefined) {
      return this._engine._executeDetectText(source, this._options);
    }
    if (this._rgb !== null) {
      return this._engine._executeDetectTextRgb(
        this._rgb.pixels,
        this._rgb.width,
        this._rgb.height,
        this._rgb.stride,
        this._options
      );
    }
    if (this._source === null || this._source === undefined) {
      throw new Error('Image source must be provided to detectText() or run()');
    }
    return this._engine._executeDetectText(this._source, this._options);
  }
}

/**
 * High-performance, pure-Rust OCR engine for Node.js via native N-API addons.
 */
class RustO {
  constructor(configOrToken = {}, rawInstance = null) {
    if (configOrToken === INTERNAL_RAW_TOKEN) {
      this._raw = rawInstance;
      return;
    }
    const json = typeof configOrToken === 'string' ? configOrToken : JSON.stringify(configOrToken ?? {});
    this._raw = RawRustO.initialize(json);
  }

  /**
   * Initializes the OCR engine with configuration options or a JSON configuration string.
   *
   * @param {import('./index').InitializeConfig | string} [config] - Initialization options or JSON string.
   * @returns {RustO} Initialized engine instance.
   */
  static initialize(config = {}) {
    const json = typeof config === 'string' ? config : JSON.stringify(config ?? {});
    const raw = RawRustO.initialize(json);
    return new RustO(INTERNAL_RAW_TOKEN, raw);
  }

  /**
   * Downloads model assets for a specific OCR pipeline ahead of time.
   *
   * @param {string} [model] - Canonical model name (e.g. 'indraocr', 'ppocrv6-tiny', 'ppocrv5'). Defaults to 'indraocr'.
   * @param {string} [lang] - Optional language code.
   * @param {string} [cacheDir] - Optional custom cache directory.
   * @returns {string[]} Array of downloaded local file paths.
   */
  static downloadModel(model, lang, cacheDir) {
    return RawRustO.downloadModel(model || null, lang || null, cacheDir || null);
  }

  /**
   * Returns the underlying RustO engine version.
   *
   * @returns {string} Version string.
   */
  static version() {
    return typeof RawRustO.version === 'function' ? RawRustO.version() : '0.4.0';
  }

  /**
   * Checks whether the given engine instance is ready for inference.
   *
   * @param {RustO} [engine] - Optional engine instance.
   * @returns {boolean} True if ready.
   */
  static isReady(engine) {
    return engine ? engine.isReady() : false;
  }


  _ensureOpen() {
    if (!this._raw) {
      throw new Error('RustO engine instance is closed.');
    }
    return this._raw;
  }

  /**
   * Detects text in an image source (file path, buffer, or source object) with runtime options,
   * or starts a fluent builder when options are omitted.
   *
   * @param {import('./index').ImageSource} [source] - Image source.
   * @param {import('./index').OcrRunOptions | string} [options] - Runtime options or JSON string.
   * @returns {DetectTextBuilder | import('./index').TextResult[] | string} Fluent builder or OCR results.
   */
  detectText(source, options) {
    if (arguments.length === 0) {
      return new DetectTextBuilder(this);
    }
    if (options === undefined) {
      return new DetectTextBuilder(this, source);
    }
    return this._executeDetectText(source, options);
  }

  /**
   * Detects text from a raw interleaved 24-bit RGB pixel buffer.
   *
   * @param {Buffer | Uint8Array} pixels - Raw interleaved RGB8 pixel buffer.
   * @param {number} width - Image width in pixels.
   * @param {number} height - Image height in pixels.
   * @param {number | null} [stride] - Optional row stride in bytes (defaults to width * 3).
   * @param {import('./index').OcrRunOptions | string} [options] - Optional runtime options or JSON string.
   * @returns {import('./index').TextResult[] | string} Structured OCR results or spatial text layout.
   */
  detectTextRgb(pixels, width, height, stride, options) {
    return this._executeDetectTextRgb(pixels, width, height, stride, options);
  }

  _executeDetectText(source, options) {
    if (typeof source === 'string') {
      return this._detectTextFile(source, options);
    }
    if (Buffer.isBuffer(source) || source instanceof Uint8Array) {
      return this._detectTextData(source, options);
    }
    if (source && typeof source === 'object') {
      if (typeof source.path === 'string') {
        return this._detectTextFile(source.path, options);
      }
      if (typeof source.uri === 'string') {
        const filePath = source.uri.replace(/^file:\/\//, '');
        return this._detectTextFile(filePath, options);
      }
      if (source.data) {
        return this._detectTextData(source.data, options);
      }
      if (source.bytes) {
        return this._detectTextData(source.bytes, options);
      }
    }
    throw new TypeError('Invalid image source: expected file path string, Buffer/Uint8Array, or source object.');
  }

  _detectTextFile(path, options) {
    const raw = this._ensureOpen();
    const opts = typeof options === 'string' ? options : (options ? JSON.stringify(options) : null);
    const rawResult = raw.detectTextFile(path, opts);
    return JSON.parse(rawResult);
  }

  _detectTextData(data, options) {
    const raw = this._ensureOpen();
    const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data);
    const opts = typeof options === 'string' ? options : (options ? JSON.stringify(options) : null);
    const rawResult = raw.detectTextData(buffer, opts);
    return JSON.parse(rawResult);
  }

  _executeDetectTextRgb(pixels, width, height, stride, options) {
    const raw = this._ensureOpen();
    const buffer = Buffer.isBuffer(pixels) ? pixels : Buffer.from(pixels);
    const effectiveStride = stride === undefined || stride === null ? null : stride;
    const opts = typeof options === 'string' ? options : (options ? JSON.stringify(options) : null);
    const rawResult = raw.detectTextRgb(buffer, width, height, effectiveStride, opts);
    return JSON.parse(rawResult);
  }

  /**
   * Releases the underlying native RustO engine reference.
   */
  close() {
    this._raw = null;
  }

  /**
   * Checks whether this engine instance is ready for inference.
   *
   * @returns {boolean} True if ready.
   */
  isReady() {
    return this._raw !== null && (typeof this._raw.isReady === 'function' ? this._raw.isReady() : true);
  }

  /**
   * Returns the underlying RustO engine version.
   *
   * @returns {string} Version string.
   */
  version() {
    return this._ensureOpen().version();
  }
}

/**
 * Returns the underlying RustO engine version string.
 *
 * @returns {string}
 */
function version() {
  return RustO.version();
}

/**
 * Checks whether an engine instance is ready for inference.
 *
 * @param {RustO} engine - Engine instance.
 * @returns {boolean}
 */
function isReady(engine) {
  return RustO.isReady(engine);
}

module.exports = { RustO, DetectTextBuilder, version, isReady };

