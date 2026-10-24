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

class RustO {
  constructor(raw) {
    this._raw = raw;
  }

  static initialize(config) {
    const json = typeof config === 'string' ? config : JSON.stringify(config);
    const raw = RawRustO.initialize(json);
    return new RustO(raw);
  }

  detectTextFile(path, options) {
    const opts = typeof options === 'string' ? options : (options ? JSON.stringify(options) : null);
    const rawResult = this._raw.detectTextFile(path, opts);
    try {
      return JSON.parse(rawResult);
    } catch {
      return rawResult;
    }
  }

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

  detectTextSpatial(path, options) {
    const opts = typeof options === 'string' ? options : (options ? JSON.stringify(options) : null);
    return this._raw.detectTextSpatial(path, opts);
  }

  version() {
    return this._raw.version();
  }
}

module.exports = { RustO };
