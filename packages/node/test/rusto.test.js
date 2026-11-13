const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');
const { RustO } = require('../index');

test('RustO N-API class is exported', () => {
  assert.strictEqual(typeof RustO, 'function');
  assert.strictEqual(typeof RustO.initialize, 'function');
  assert.strictEqual(typeof RustO.downloadModel, 'function');
  assert.strictEqual(typeof RustO.version, 'function');
  assert.strictEqual(typeof RustO.isReady, 'function');
  assert.strictEqual(typeof RustO.version(), 'string');
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

  const engine = new RustO({
    detection: { model_path: detPath },
    recognition: { model_path: recPath, dict_path: dictPath }
  });

  const results = engine.detectText(imgPath)
    .withRecognition({ batchSize: 8 })
    .withPreDetection({ targetMaxSide: 960, cropPaddingX: 0.05, cropPaddingY: 0.08 })
    .asLines()
    .run();
  assert.ok(Array.isArray(results), 'Expected results to be an array');
  assert.ok(results.length > 0, 'Expected detected text lines');
  assert.ok(results.some((r) => r.text.includes('INVOICE')), 'Expected to find INVOICE in detected text');

  const spatial = engine.detectText(imgPath)
    .withRecognition({ batchSize: 8 })
    .asSpatial()
    .run();
  assert.strictEqual(typeof spatial, 'string');
  assert.ok(spatial.includes('INVOICE'), 'Expected spatial output to contain INVOICE');

  engine.close();
  assert.throws(() => engine.version(), /closed/);
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

  const results = engine.detectText(imgPath)
    .withRecognition({ batchSize: 8 })
    .withPostRecognition({ scoreThreshold: 0.3 })
    .asLines()
    .run();
  assert.ok(Array.isArray(results), 'Expected results to be an array');
  assert.ok(results.length > 0, 'Expected detected text lines');
  const joined = results.map((r) => r.text).join(' ');
  assert.ok(
    joined.includes('UNITED') || joined.includes('PASSPORT') || joined.includes('BRITAIN'),
    `Expected passport text, got: ${joined}`
  );
  engine.close();
});

test('RustO performs real OCR on Issue 25 chronam-ad-page.png with layout', (t) => {
  const detPath = path.resolve(__dirname, '../../../models/PPOCR_v6_tiny/det.rten');
  const recPath = path.resolve(__dirname, '../../../models/PPOCR_v6_tiny/rec.rten');
  const dictPath = path.resolve(__dirname, '../../../models/PPOCR_v6_tiny/dict.txt');
  const layoutPath = path.resolve(__dirname, '../../../models/PPOCR_v6_tiny/layout_cdla.onnx');
  const imgPath = path.resolve(__dirname, '../../../tests/fixtures/chronam-ad-page.png');

  if (!fs.existsSync(detPath) || !fs.existsSync(imgPath)) {
    t.skip('Model files or test image not found, skipping');
    return;
  }

  const engine = RustO.initialize({
    detection: { model_path: detPath },
    recognition: { model_path: recPath, dict_path: dictPath },
    layout: { model_path: layoutPath }
  });

  const results = engine.detectText(imgPath, {
    layoutReading: true
  });
  assert.strictEqual(results.length, 238, 'Expected 238 lines');
  assert.strictEqual(results[0].text, 'All Orders in our');
  assert.strictEqual(typeof results[0].region_id, 'number');
  assert.strictEqual(typeof results[0].layout_type, 'string');
  engine.close();
});

test('RustO fluent builder chaining detectText().with...().asLines().run()', (t) => {
  const detPath = path.resolve(__dirname, '../../../models/PPOCR_v6_tiny/det.rten');
  const recPath = path.resolve(__dirname, '../../../models/PPOCR_v6_tiny/rec.rten');
  const dictPath = path.resolve(__dirname, '../../../models/PPOCR_v6_tiny/dict.txt');
  const layoutPath = path.resolve(__dirname, '../../../models/PPOCR_v6_tiny/layout_cdla.onnx');
  const imgPath = path.resolve(__dirname, '../../../tests/fixtures/chronam-ad-page.png');

  if (!fs.existsSync(detPath) || !fs.existsSync(imgPath)) {
    t.skip('Model files or test image not found, skipping');
    return;
  }

  const engine = RustO.initialize({
    detection: { model_path: detPath },
    recognition: { model_path: recPath, dict_path: dictPath },
    layout: { model_path: layoutPath }
  });

  const results = engine.detectText(imgPath)
    .withLayoutReading(true)
    .withPreDetection({ descreenStrength: 0.1 })
    .withRecognition({ batchSize: 6, workers: 2, maxWorkers: 4 })
    .withPostRecognition({ scoreThreshold: 0.5 })
    .asLines()
    .run();

  assert.ok(Array.isArray(results), 'Expected array results');
  assert.ok(results.length > 0, 'Expected detected lines');
  assert.ok(results[0].text.startsWith('All Orders in'));
  assert.strictEqual(typeof results[0].region_id, 'number');

  const spatial = engine.detectText()
    .image(imgPath)
    .withLayoutReading(true)
    .asSpatial()
    .run();

  assert.strictEqual(typeof spatial, 'string');
  assert.ok(spatial.includes('Orders'));
  engine.close();
});
