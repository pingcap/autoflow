import { authenticationHeaders, handleResponse, requestUrl } from '@/lib/request';
import { z } from 'zod';

const configurationSchema = z.object({
  host: z.string(),
  public_key: z.string(),
  configured: z.boolean(),
  revision: z.string(),
  migration_required: z.boolean(),
});

export type LangfuseConfiguration = z.infer<typeof configurationSchema>;
export interface LangfuseUpdate {
  host: string;
  public_key: string;
  revision: string;
  secret_key?: string;
  clear_secret?: boolean;
}

export async function getLangfuseConfiguration () {
  return fetch(requestUrl('/api/v1/admin/langfuse'), {
    headers: await authenticationHeaders(),
    cache: 'no-store',
  }).then(handleResponse(configurationSchema));
}

export async function updateLangfuseConfiguration (configuration: LangfuseUpdate) {
  return fetch(requestUrl('/api/v1/admin/langfuse'), {
    method: 'PUT',
    headers: { ...await authenticationHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify(configuration),
  }).then(handleResponse(configurationSchema));
}
