import { safeLoginCallback } from './safe-login-callback';

const origin = 'https://chat.example.com';

test.each([
  undefined, '', 'https://attacker.example/collect', '//attacker.example',
  'javascript:alert(1)', 'data:text/html,hello', '/\\attacker.example',
  '/%2f/attacker.example', '/%5cattacker.example', '/%zz',
  'https://user:password@chat.example.com/path', '/auth/login?next=/admin',
  '/%61uth/login', '\nhttps://attacker.example', 'a'.repeat(2049),
])('rejects unsafe login callback %s', url => {
  expect(safeLoginCallback(url, origin)).toBe('/');
});

test.each(['/chats/123?q=ok#latest', `${origin}/chats/123?q=ok#latest`])('preserves a local route', url => {
  expect(safeLoginCallback(url, origin)).toBe('/chats/123?q=ok#latest');
});
