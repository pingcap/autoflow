/** @jest-environment node */
import { NextRequest } from 'next/server';
import { jest } from '@jest/globals';

jest.unstable_mockModule('../../../lib/request', () => ({ BASE_URL: 'http://backend.internal' }));

const { POST } = await import('./route');
const mockFetch = jest.fn<typeof fetch>();
let originalFetch: typeof fetch;
beforeEach(() => {
  originalFetch = global.fetch;
  mockFetch.mockReset().mockResolvedValue(new Response('{}', { status: 200 }));
  global.fetch = mockFetch;
  jest.spyOn(console, 'log').mockImplementation(() => {});
});
afterEach(() => {
  global.fetch = originalFetch;
  jest.restoreAllMocks();
});

test('same-origin browser requests retain cookies and receive explicit upstream CSRF proof', async () => {
  await POST(new NextRequest('https://app.example/api/v1/chats', { method: 'POST', headers: { Origin: 'https://app.example', Referer: 'https://app.example/?private=value', Cookie: 'session=test-session' } }));
  const [, init] = mockFetch.mock.calls[0];
  const headers = init!.headers as Headers;
  expect(headers.get('Origin')).toBeNull();
  expect(headers.get('Referer')).toBeNull();
  expect(headers.get('X-CSRF-Guard')).toBe('1');
  expect(headers.get('Cookie')).toBe('session=test-session');
});

test('cross-origin provenance is preserved for backend verification', async () => {
  await POST(new NextRequest('https://app.example/api/v1/chats', { method: 'POST', headers: { Origin: 'https://evil.example', Cookie: 'bid=test', 'X-CSRF-Guard': '1' } }));
  const [, init] = mockFetch.mock.calls[0];
  expect((init!.headers as Headers).get('Origin')).toBe('https://evil.example');
});

test('originless requests do not receive synthesized CSRF proof', async () => {
  await POST(new NextRequest('https://app.example/api/v1/chats', { method: 'POST', headers: { Cookie: 'bid=test' } }));
  const [, init] = mockFetch.mock.calls[0];
  expect((init!.headers as Headers).get('X-CSRF-Guard')).toBeNull();
});


test('upstream failures do not expose query values or exception messages', async () => {
  mockFetch.mockRejectedValue(new Error('private-provider-token'));
  const log = jest.spyOn(console, 'error').mockImplementation(() => {});
  const response = await POST(new NextRequest('https://app.example/api/resource?token=private-query', { method: 'POST' }));
  expect(response.status).toBe(502);
  expect(await response.text()).toBe('{"detail":"Upstream service unavailable"}');
  expect(JSON.stringify(log.mock.calls)).not.toContain('private-');
});
