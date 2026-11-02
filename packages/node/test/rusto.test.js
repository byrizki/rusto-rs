const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');
const { RustO } = require('../index');

test('RustO N-API class is exported', () => {
  assert.strictEqual(typeof RustO, 'function');
  assert.strictEqual(typeof RustO.initialize, 'function');
  assert.strictEqual(typeof RustO.downloadModel, 'function');
  assert.strictEqual(typeof RustO.downloadAllModels, 'function');
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

  const results = engine.detectTextFile(imgPath, {
    recBatchNum: 8,
    optimization: { enabled: true, targetMaxSide: 960, cropPaddingX: 0.05, cropPaddingY: 0.08 }
  });
  assert.ok(Array.isArray(results), 'Expected results to be an array');
  assert.ok(results.length > 0, 'Expected detected text lines');
  assert.ok(results.some((r) => r.text.includes('INVOICE')), 'Expected to find INVOICE in detected text');

  const spatial = engine.detectTextSpatial(imgPath, { recBatchNum: 8 });
  assert.strictEqual(typeof spatial, 'string');
  assert.ok(spatial.includes('INVOICE'), 'Expected spatial output to contain INVOICE');
});

test('RustO performs real OCR with Indra OCR on idcard.jpg', () => {
  const detPath = path.resolve(__dirname, '../../../models/indra/det.rten');
  const recPath = path.resolve(__dirname, '../../../models/indra/rec.rten');
  const dictPath = path.resolve(__dirname, '../../../models/indra/meta.json');
  const imgPath = path.resolve(__dirname, '../../../models/test_images/idcard.jpg');

  if (!fs.existsSync(detPath) || !fs.existsSync(imgPath)) {
    return;
  }

  const engine = RustO.initialize({
    detection: { model_path: detPath },
    recognition: { model_path: recPath, dict_path: dictPath }
  });

  const results = engine.detectTextFile(imgPath, {
    recBatchNum: 8,
    textScore: 0.3
  });
  assert.ok(Array.isArray(results), 'Expected results to be an array');
  assert.ok(results.length > 0, 'Expected detected text lines');
  const joined = results.map((r) => r.text).join(' ');
  assert.ok(
    joined.includes('UNITED') || joined.includes('PASSPORT') || joined.includes('BRITAIN'),
    `Expected passport text, got: ${joined}`
  );
});
