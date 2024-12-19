import { type ProviderOption, providerOptionSchema } from '@/api/providers';
import { authenticationHeaders, handleNullableResponse, handleResponse, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { z, type ZodType, type ZodTypeDef } from 'zod';

export interface EmbeddingModelSummary {
  id: number;
  name: string;
  provider: string;
  model: string;
  vector_dimension: number,
  is_default: boolean
}

export interface EmbeddingModel extends EmbeddingModelSummary {
  config?: any;
  created_at?: Date | null;
  updated_at?: Date | null;
}

export interface EmbeddingModelOption extends ProviderOption {
  default_embedding_model: string;
  embedding_model_description: string;
}

export interface CreateEmbeddingModel {
  name: string;
  provider: string;
  model: string;
  config?: any;
  credentials: string | object;
}

export const embeddingModelSummarySchema = z.object({
  id: z.number(),
  name: z.string(),
  provider: z.string(),
  model: z.string(),
  vector_dimension: z.number(),
  is_default: z.boolean(),
}) satisfies ZodType<EmbeddingModelSummary, ZodTypeDef, any>;

const embeddingModelSchema = embeddingModelSummarySchema.extend({
  config: z.any(),
  created_at: zodJsonDate().nullable().optional(),
  updated_at: zodJsonDate().nullable().optional(),
}) satisfies ZodType<EmbeddingModel, ZodTypeDef, any>;

const embeddingModelOptionSchema = providerOptionSchema.and(z.object({
  default_embedding_model: z.string(),
  embedding_model_description: z.string(),
})) satisfies ZodType<EmbeddingModelOption, any, any>;

export async function listEmbeddingModelOptions () {
  return await fetch(requestUrl(`/api/v1/admin/embedding-models/options`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(embeddingModelOptionSchema.array()));
}

export async function getEmbeddingModel (id: number) {
  return await fetch(requestUrl(`/api/v1/admin/embedding-models/${id}`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(embeddingModelSchema));
}

export async function listEmbeddingModels (params: PageParams) {
  return await fetch(requestUrl(`/api/v1/admin/embedding-models`, params), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(embeddingModelSchema)));
}

export async function createEmbeddingModel (create: CreateEmbeddingModel) {
  return await fetch(requestUrl(`/api/v1/admin/embedding-models`), {
    method: 'POST',
    body: JSON.stringify(create),
    headers: {
      'Content-Type': 'application/json',
      ...await authenticationHeaders(),
    },
  }).then(handleResponse(embeddingModelSchema));
}

export async function testEmbeddingModel (createEmbeddingModel: CreateEmbeddingModel) {
  return await fetch(requestUrl(`/api/v1/admin/embedding-models/test`), {
    method: 'POST',
    body: JSON.stringify(createEmbeddingModel),
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

