/** @jest-environment node */
import { z } from 'zod';
import { assertValidationBudget, readBoundedResponseText, readValidatedJson } from './validation-budget';

test('patched bounded arrays reject before validating any element', () => {
  let inspected = 0;
  const schema = z.array(z.custom(() => { inspected++; return false; })).max(100);
  const result = schema.safeParse(Array(100001).fill(null));
  expect(result.success).toBe(false);
  expect(inspected).toBe(0);
  if (!result.success) expect(result.error.issues).toHaveLength(1);
});

test('unbounded response schemas cannot walk an oversized collection', async () => {
  let inspected = 0;
  const schema = z.array(z.custom(() => { inspected++; return false; }));
  const response = new Response(JSON.stringify(Array(10001).fill(null)));
  await expect(readValidatedJson(response).then(body => schema.parse(body))).rejects.toThrow('collection');
  expect(inspected).toBe(0);
});

test('byte limits use received bytes even when content-length is missing or false', async () => {
  const variants: HeadersInit[] = [{}, { 'content-length': '1' }];
  for (const headers of variants) {
    await expect(readBoundedResponseText(new Response('é'.repeat(10), { headers }), 16)).rejects.toThrow('size');
  }
});

test('oversized streams are cancelled and do not continue consuming chunks', async () => {
  let cancelled = false;
  let pulled = 0;
  const body = new ReadableStream<Uint8Array>({
    pull (controller) { pulled++; controller.enqueue(new Uint8Array(20)); },
    cancel () { cancelled = true; },
  });
  await expect(readBoundedResponseText(new Response(body), 16)).rejects.toThrow('size');
  expect(cancelled).toBe(true);
  expect(pulled).toBeLessThanOrEqual(2);
});

test('nested arrays and objects have a shared node and depth budget', () => {
  expect(() => assertValidationBudget(Array.from({ length: 101 }, () => Array(1000).fill(null)))).toThrow('complexity');
  let deep: unknown = 'end';
  for (let i = 0; i < 66; i++) deep = { next: deep };
  expect(() => assertValidationBudget(deep)).toThrow('complexity');
  expect(() => assertValidationBudget(Object.fromEntries(Array.from({ length: 10001 }, (_, i) => [String(i), i])))).toThrow('collection');
});

test('ordinary pages, metadata, Unicode and document text remain intact', async () => {
  const value = { items: Array.from({ length: 100 }, (_, id) => ({ id, meta: { labels: ['a', 'b'] }, content: '文档\n' })), content: 'x'.repeat(1024 * 1024) };
  const parsed = await readValidatedJson(new Response(JSON.stringify(value)));
  expect(parsed).toEqual(value);
});
