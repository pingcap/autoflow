import { type ProviderOption, providerOptionSchema } from '@/api/providers';
import { authenticationHeaders, handleErrors, handleResponse, type Page, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { z, type ZodType, type ZodTypeDef } from 'zod';

export interface Reranker {
  id: number;
  name: string;
  provider: string;
  model: string;
  top_n: number;
  config?: any;
  is_default: boolean;
  created_at: Date | null;
  updated_at: Date | null;
}

export interface RerankerOption extends ProviderOption {
  default_reranker_model: string;
  reranker_model_description: string;
  default_top_n: number;
}

export interface CreateReranker {
  name: string;
  provider: string;
  model: string;
  config?: any;
  top_n: number;
  is_default?: boolean;
  credentials: string | object;
}

const rerankerSchema = z.object({
  id: z.number(),
  name: z.string(),
  provider: z.string(),
  model: z.string(),
  top_n: z.number(),
  config: z.any(),
  is_default: z.boolean(),
  created_at: zodJsonDate().nullable(),
  updated_at: zodJsonDate().nullable(),
}) satisfies ZodType<Reranker, ZodTypeDef, any>;

const rerankerOptionSchema = providerOptionSchema.and(z.object({
  default_top_n: z.number(),
  default_reranker_model: z.string(),
  reranker_model_description: z.string(),
})) satisfies ZodType<RerankerOption, any, any>;

export async function listRerankerOptions () {
  return await fetch(requestUrl(`/api/v1/admin/reranker-models/providers/options`), {
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(rerankerOptionSchema.array()));
}

export async function listRerankers ({ page = 1, size = 10 }: PageParams = {}): Promise<Page<Reranker>> {
  return await fetch(requestUrl('/api/v1/admin/reranker-models', { page, size }), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(rerankerSchema)));
}

export async function getReranker (id: number): Promise<Reranker> {
  return await fetch(requestUrl(`/api/v1/admin/reranker-models/${id}`), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(rerankerSchema));
}

export async function createReranker (create: CreateReranker) {
  return await fetch(requestUrl(`/api/v1/admin/reranker-models`), {
    method: 'POST',
    body: JSON.stringify(create),
    headers: {
      'Content-Type': 'application/json',
      ...await authenticationHeaders(),
    },
  }).then(handleResponse(rerankerSchema));
}

export async function deleteReranker (id: number) {
  await fetch(requestUrl(`/api/v1/admin/reranker-models/${id}`), {
    method: 'DELETE',
    headers: await authenticationHeaders(),
  }).then(handleErrors);
}

export async function testReranker (createReranker: CreateReranker) {
  return await fetch(requestUrl(`/api/v1/admin/reranker-models/test`), {
    method: 'POST',
    body: JSON.stringify(createReranker),
    headers: {
      'Content-Type': 'application/json',
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(z.object({
      success: z.boolean(),
      error: z.string().optional(),
    })));
}
