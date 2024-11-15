import { type ProviderOption, providerOptionSchema } from '@/api/providers';
import { authenticationHeaders, handleErrors, handleResponse, type Page, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { z, type ZodType, type ZodTypeDef } from 'zod';

export interface LLMSummary {
  id: number;
  name: string;
  provider: string;
  model: string;
  is_default: boolean;
}

export interface LLM extends LLMSummary {
  config?: any;
  created_at: Date | null;
  updated_at: Date | null;
}

export interface LlmOption extends ProviderOption {
  default_llm_model: string;
  llm_model_description: string;
}

export interface CreateLLM {
  name: string;
  provider: string;
  model: string;
  config?: any;
  is_default?: boolean;
  credentials: string | object;
}

export const llmSummarySchema = z.object({
  id: z.number(),
  name: z.string(),
  provider: z.string(),
  model: z.string(),
  is_default: z.boolean(),
}) satisfies ZodType<LLMSummary, ZodTypeDef, any>;

const llmSchema = llmSummarySchema.extend({
  config: z.any(),
  created_at: zodJsonDate().nullable(),
  updated_at: zodJsonDate().nullable(),
}) satisfies ZodType<LLM, ZodTypeDef, any>;

const llmOptionSchema = providerOptionSchema.and(z.object({
  default_llm_model: z.string(),
  llm_model_description: z.string(),
})) satisfies ZodType<LlmOption, any, any>;

export async function listLlmOptions () {
  return await fetch(requestUrl(`/api/v1/admin/llms/options`), {
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(llmOptionSchema.array()));
}

export async function listLlms ({ page = 1, size = 10 }: PageParams = {}): Promise<Page<LLM>> {
  return await fetch(requestUrl('/api/v1/admin/llms', { page, size }), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(llmSchema)));
}

export async function getLlm (id: number): Promise<LLM> {
  return await fetch(requestUrl(`/api/v1/admin/llms/${id}`), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(llmSchema));
}

export async function createLlm (create: CreateLLM) {
  return await fetch(requestUrl(`/api/v1/admin/llms`), {
    method: 'POST',
    body: JSON.stringify(create),
    headers: {
      'Content-Type': 'application/json',
      ...await authenticationHeaders(),
    },
  }).then(handleResponse(llmSchema));
}

export async function deleteLlm (id: number) {
  await fetch(requestUrl(`/api/v1/admin/llms/${id}`), {
    method: 'DELETE',
    headers: await authenticationHeaders(),
  }).then(handleErrors);
}

export async function testLlm (createLLM: CreateLLM) {
  return await fetch(requestUrl(`/api/v1/admin/llms/test`), {
    method: 'POST',
    body: JSON.stringify(createLLM),
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
