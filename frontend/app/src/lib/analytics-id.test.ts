import { safeAnalyticsId } from './analytics-id';

test.each(['G-ABC1234567', 'UA-12345678-1'])('accepts a measurement ID', id => {
  expect(safeAnalyticsId(id)).toBe(id);
});

test.each([null, {}, '', "G-ABC123');alert(1)//", '</script><script>alert(1)</script>', 'G-ABC123\n', 'https://example.com'])('suppresses unsafe persisted analytics values', value => {
  expect(safeAnalyticsId(value)).toBe('');
});
