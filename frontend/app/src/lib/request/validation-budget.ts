// A 10 MiB uploaded text document can expand sixfold when JSON-escaped.
// Keep ordinary document previews while bounding response decoding and validation.
export const MAX_JSON_RESPONSE_BYTES = 64 * 1024 * 1024;
export const MAX_COLLECTION_ITEMS = 10000;
export const MAX_VALIDATION_NODES = 100000;
export const MAX_VALIDATION_DEPTH = 64;

export async function readBoundedResponseText (response: Response, maxBytes = MAX_JSON_RESPONSE_BYTES) {
  const reader = response.body?.getReader();
  if (!reader) return '';
  let complete = false;
  try {
    const declared = response.headers.get('content-length');
    if (declared && /^\d+$/.test(declared) && Number(declared) > maxBytes) {
      throw new Error('Response exceeds the supported size.');
    }
    const decoder = new TextDecoder('utf-8', { fatal: true });
    const parts: string[] = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      received += value.byteLength;
      if (received > maxBytes) throw new Error('Response exceeds the supported size.');
      parts.push(decoder.decode(value, { stream: true }));
    }
    parts.push(decoder.decode());
    complete = true;
    return parts.join('');
  } finally {
    if (!complete) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export function assertValidationBudget (value: unknown) {
  const pending: { value: unknown, depth: number }[] = [{ value, depth: 0 }];
  let visited = 0;
  while (pending.length) {
    const { value, depth } = pending.pop()!;
    if (++visited > MAX_VALIDATION_NODES || depth > MAX_VALIDATION_DEPTH) {
      throw new Error('Response structure exceeds the supported complexity.');
    }
    if (value === null || typeof value !== 'object') continue;
    if (Array.isArray(value)) {
      if (value.length > MAX_COLLECTION_ITEMS) throw new Error('Response collection is too large.');
      for (const item of value) pending.push({ value: item, depth: depth + 1 });
    } else {
      let keys = 0;
      for (const key in value) {
        if (!Object.hasOwn(value, key)) continue;
        if (++keys > MAX_COLLECTION_ITEMS) throw new Error('Response collection is too large.');
        pending.push({ value: (value as Record<string, unknown>)[key], depth: depth + 1 });
      }
    }
    if (visited + pending.length > MAX_VALIDATION_NODES) {
      throw new Error('Response structure exceeds the supported complexity.');
    }
  }
}

export async function readValidatedJson (response: Response) {
  const body: unknown = JSON.parse(await readBoundedResponseText(response));
  assertValidationBudget(body);
  return body;
}
