import { authenticationHeaders, handleErrors, handleResponse, type Page, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { z, type ZodType, type ZodTypeDef } from 'zod';

export type ApiKeyScope = 'retrieve:chunks' | 'retrieve:graph' | 'chat:write';

export interface ApiKey {
  created_at: Date;
  updated_at: Date;
  description: string;
  api_key_display: string;
  is_active: boolean;
  expires_at: Date | null;
  scopes: ApiKeyScope[] | null;
  knowledge_base_ids: number[] | null;
  chat_engine_ids: number[] | null;
  user_id: string;
  id: number;
}

export interface CreateApiKey {
  description: string;
  scopes: ApiKeyScope[];
  knowledge_base_ids: number[];
  chat_engine_ids: number[];
  expires_in_days: number;
}

export interface CreateApiKeyResponse {
  api_key: string;
}

const apiKeySchema = z.object({
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
  description: z.string(),
  api_key_display: z.string(),
  is_active: z.boolean(),
  expires_at: zodJsonDate().nullable(),
  scopes: z.array(z.enum(['retrieve:chunks', 'retrieve:graph', 'chat:write'])).nullable(),
  knowledge_base_ids: z.array(z.number()).nullable(),
  chat_engine_ids: z.array(z.number()).nullable(),
  user_id: z.string(),
  id: z.number(),
}) satisfies ZodType<ApiKey, ZodTypeDef, any>;

const createApiKeyResponseSchema = z.object({
  api_key: z.string(),
}) satisfies ZodType<CreateApiKeyResponse>;

export async function listApiKeys ({ page = 1, size = 10 }: PageParams = {}): Promise<Page<ApiKey>> {
  return await fetch(requestUrl('/api/v1/api-keys', { page, size }), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(apiKeySchema)));
}

export async function createApiKey (create: CreateApiKey): Promise<CreateApiKeyResponse> {
  return await fetch(requestUrl('/api/v1/api-keys'), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(create),
  }).then(handleResponse(createApiKeyResponseSchema));
}

export async function deleteApiKey (id: number): Promise<void> {
  await fetch(requestUrl(`/api/v1/api-keys/${id}`), {
    method: 'DELETE',
    headers: {
      ...await authenticationHeaders(),
    },
  }).then(handleErrors);
}

