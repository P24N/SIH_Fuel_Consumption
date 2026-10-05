const test = require('node:test');
const assert = require('node:assert/strict');
const {parseCSV, toCSV} = require('../public/app.js');
test('CSV round trips quoted commas, quotes, and newlines', () => {
  const rows = [{Vessel: 'Ship "One", North\nFleet', Capacity: '50000'}];
  assert.deepEqual(parseCSV(toCSV(rows)), rows);
});
test('CSV rejects malformed headers, quotes, and row widths', () => {
  for (const csv of ['a,a\n1,2', 'a,b\n1', 'a\n"unterminated']) assert.throws(() => parseCSV(csv));
});
test('CSV escapes spreadsheet formulas', () => {
  assert.equal(parseCSV(toCSV([{Vessel: '=HYPERLINK("bad")'}]))[0].Vessel[0], "'");
});
