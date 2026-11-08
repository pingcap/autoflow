import type { Response } from '@playwright/test';

class BodyReadTimeout extends Error {}

function bodyFailure (error: unknown) {
  if (error instanceof BodyReadTimeout) return 'body-read-timeout';
  const message = error instanceof Error ? error.message : '';
  const code = message.match(/net::ERR_[A-Z_]+/)?.[0];
  if (code) return code;
  if (message.includes('No resource with given identifier')) return 'body-unavailable';
  if (message.includes('Target closed') || message.includes('Target page, context or browser has been closed')) return 'page-closed';
  return 'body-read-failed';
}

export async function readModelVerification (response: Response) {
  // Distinguish failed browser body retrieval from malformed server JSON.
  // Only fixed classifications and byte counts may appear in test output.
  let timer: ReturnType<typeof setTimeout> | undefined;
  let bytes: Buffer;
  try {
    bytes = await Promise.race([
      (async () => {
        const finished = await response.finished();
        if (finished) throw finished;
        return response.body();
      })(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new BodyReadTimeout()), 10000);
      }),
    ]);
  } catch (error) {
    throw new Error(`LLM verification transport failed (${bodyFailure(error)})`);
  } finally {
    clearTimeout(timer);
  }
  let result: unknown;
  try {
    result = JSON.parse(bytes.toString('utf-8'));
  } catch {
    throw new Error(`LLM verification returned invalid JSON (${bytes.length} bytes)`);
  }
  if (!result || typeof result !== 'object' || !('success' in result) || typeof result.success !== 'boolean') {
    throw new Error('LLM verification returned an invalid result shape');
  }
  const stage = 'stage' in result && typeof result.stage === 'string' && ['configuration', 'llm', 'dspy'].includes(result.stage)
    ? result.stage : 'unknown';
  const errorType = 'error_type' in result && typeof result.error_type === 'string' && /^[A-Za-z_][A-Za-z0-9_]{0,79}$/.test(result.error_type)
    ? result.error_type : 'unknown';
  return { success: result.success, stage, errorType };
}
