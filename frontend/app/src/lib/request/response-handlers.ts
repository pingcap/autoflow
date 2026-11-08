import { normalizeServerErrors } from '@/lib/request/index';
import { z, ZodType } from 'zod';
import { readBoundedResponseText, readValidatedJson } from './validation-budget';

export async function handleErrors (responseOrPromise: Response | PromiseLike<Response>): Promise<Response> {
  const response = await responseOrPromise;
  if (response.ok) {
    return response;
  }

  try {
    const body = await readBoundedResponseText(response, 65536);
    let parsed: unknown;
    try {
      parsed = JSON.parse(body);
    } catch {
      parsed = body;
    }
    return Promise.reject(normalizeServerErrors(response, parsed));
  } catch {
    return Promise.reject(normalizeServerErrors(response, `${response.status} ${response.statusText}`));
  }
}

export function handleResponse<S extends ZodType> (schema: S): ((responseOrPromise: Response | PromiseLike<Response>) => Promise<z.infer<S>>) {
  return async (responseOrPromise) => {
    const response = await Promise.resolve(responseOrPromise).then(handleErrors);
    const body = await readValidatedJson(response);

    try {
      return schema.parse(body);
    } catch (e) {
      console.error('Response validation failed; check frontend and backend versions.');
      throw e;
    }
  };
}

export function handleNullableResponse<S extends ZodType> (schema: S): ((responseOrPromise: Response | PromiseLike<Response>) => Promise<z.infer<S> | null>) {
  return async (responseOrPromise) => {
    const response = await responseOrPromise;

    if (response.status === 404) {
      return null;
    }

    await handleErrors(response);
    const body = await readValidatedJson(response);

    try {
      return schema.parse(body);
    } catch (e) {
      console.error('Response validation failed; check frontend and backend versions.');
      throw e;
    }
  };
}
