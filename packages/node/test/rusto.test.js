const test = require('node:test');
const assert = require('node:assert');
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
