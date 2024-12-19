import { authenticationHeaders, handleResponse, requestUrl } from '@/lib/request';

import { z } from 'zod';

export interface RequiredBootstrapStatus {
  default_llm: boolean;
  default_embedding_model: boolean;
  default_chat_engine: boolean;
  knowledge_base: boolean;
}

export interface OptionalBootstrapStatus {
  langfuse: boolean;
  default_reranker: boolean;
}

export interface NeedMigrationStatus {
  chat_engines_without_kb_configured?: number[];
}

export interface BootstrapStatus {
  required: RequiredBootstrapStatus;
  optional: OptionalBootstrapStatus;
  need_migration: NeedMigrationStatus;
}

const requiredBootstrapStatusSchema = z.object({
  default_llm: z.boolean(),
  default_embedding_model: z.boolean(),
  default_chat_engine: z.boolean(),
  knowledge_base: z.boolean(),
});

const optionalBootstrapStatusSchema = z.object({
  langfuse: z.boolean(),
  default_reranker: z.boolean(),
});

const needMigrationStatusSchema = z.object({
  chat_engines_without_kb_configured: z.number().array().optional(),
});

const bootstrapStatusSchema = z.object({
  required: requiredBootstrapStatusSchema,
  optional: optionalBootstrapStatusSchema,
  need_migration: needMigrationStatusSchema,
});

export async function getBootstrapStatus (): Promise<BootstrapStatus> {
  return await fetch(requestUrl(`/api/v1/system/bootstrap-status`), {
    headers: {
      ...await authenticationHeaders(),
    },
    credentials: 'include',
  }).then(handleResponse(bootstrapStatusSchema));
}

export function isBootstrapStatusPassed (bootstrapStatus: BootstrapStatus): boolean {
  return Object.values(bootstrapStatus.required).reduce((res, flag) => res && flag, true);
}
