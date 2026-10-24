const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');
const { RustO } = require('../index');

test('RustO N-API class is exported', () => {
  assert.strictEqual(typeof RustO, 'function');
  assert.strictEqual(typeof RustO.initialize, 'function');
});

test('RustO rejects invalid initialization JSON', () => {
  assert.throws(() => {
    RustO.initialize('invalid-non-json');
  }, /Invalid config JSON/);
});

test('RustO performs real OCR on example1.png', () => {
  const detPath = path.resolve(__dirname, '../../../models/PPOCR_v6/det.rten');
  const recPath = path.resolve(__dirname, '../../../models/PPOCR_v6/rec.rten');
  const dictPath = path.resolve(__dirname, '../../../models/PPOCR_v6/dict.txt');
  const imgPath = path.resolve(__dirname, '../../../models/test_images/example1.png');

  if (!fs.existsSync(detPath) || !fs.existsSync(imgPath)) {
    return;
  }

  const engine = RustO.initialize({
    detection: { model_path: detPath },
    recognition: { model_path: recPath, dict_path: dictPath }
  });

  const results = engine.detectTextFile(imgPath);
  assert.ok(Array.isArray(results), 'Expected results to be an array');
  assert.ok(results.length > 0, 'Expected detected text lines');
  assert.ok(results.some((r) => r.text.includes('INVOICE')), 'Expected to find INVOICE in detected text');

  const spatial = engine.detectTextSpatial(imgPath);
  assert.strictEqual(typeof spatial, 'string');
  assert.ok(spatial.includes('INVOICE'), 'Expected spatial output to contain INVOICE');
});
