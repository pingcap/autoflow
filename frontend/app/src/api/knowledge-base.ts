import { type BaseCreateDatasourceParams, type CreateDatasourceSpecParams, type Datasource, type DatasourceKgIndexError, datasourceSchema, type DatasourceVectorIndexError } from '@/api/datasources';
import { documentSchema } from '@/api/documents';
import { type EmbeddingModelSummary, embeddingModelSummarySchema } from '@/api/embedding-models';
import { type LLMSummary, llmSummarySchema } from '@/api/llms';
import { type IndexProgress, indexSchema, indexStatusSchema, type IndexTotalStats, totalSchema } from '@/api/rag';
import { authenticationHeaders, handleErrors, handleResponse, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { z, type ZodType } from 'zod';

export type KnowledgeBaseIndexMethod = 'vector' | 'knowledge_graph';

export interface CreateKnowledgeBaseParams {
  name: string;
  description?: string | null;
  index_methods: KnowledgeBaseIndexMethod[];
  llm_id?: number | null;
  embedding_model_id?: number | null;
  data_sources: (BaseCreateDatasourceParams & CreateDatasourceSpecParams)[];
}

export interface UpdateKnowledgeBaseParams {
  name?: string;
  description?: string | null;
}

export interface KnowledgeBaseSummary {
  id: number;
  name: string;
  description: string | null;
  index_methods: KnowledgeBaseIndexMethod[];
  documents_total?: number;
  data_sources_total?: number;
  created_at: Date;
  updated_at: Date;
  creator: {
    id: string;
  };
}

export interface KnowledgeBase extends KnowledgeBaseSummary {
  data_sources: Datasource[];
  llm?: LLMSummary | null;
  embedding_model?: EmbeddingModelSummary | null;
  chunking_config: KnowledgeBaseChunkingConfig | null;
}

export type KnowledgeGraphIndexProgress = {
  vector_index: IndexProgress
  documents: IndexTotalStats
  chunks: IndexTotalStats
  kg_index?: IndexProgress
  entities?: IndexTotalStats
  relationships?: IndexTotalStats
}

export type KnowledgeBaseSplitterType = KnowledgeBaseChunkingSplitterRule['splitter'];

export type KnowledgeBaseChunkingSentenceSplitterConfig = {
  chunk_size: number
  chunk_overlap: number
  paragraph_separator: string
}

export type KnowledgeBaseChunkingMarkdownSplitterConfig = {
  chunk_size: number
  chunk_header_level: number
}

export type KnowledgeBaseChunkingSentenceSplitterRule = {
  splitter: 'SentenceSplitter'
  splitter_config: KnowledgeBaseChunkingSentenceSplitterConfig
}

export type KnowledgeBaseChunkingMarkdownSplitterRule = {
  splitter: 'MarkdownSplitter'
  splitter_config: KnowledgeBaseChunkingMarkdownSplitterConfig
}

export type KnowledgeBaseChunkingSplitterRule = KnowledgeBaseChunkingSentenceSplitterRule | KnowledgeBaseChunkingMarkdownSplitterRule;

export type KnowledgeBaseChunkingConfigGeneral = {
  mode: 'general'
} & KnowledgeBaseChunkingSentenceSplitterConfig;

export type KnowledgeBaseChunkingConfigAdvanced = {
  mode: 'advanced'
  rules: {
    'text/plain': KnowledgeBaseChunkingSplitterRule;
    'text/markdown': KnowledgeBaseChunkingSplitterRule
  }
}

export type KnowledgeBaseChunkingConfig = KnowledgeBaseChunkingConfigGeneral | KnowledgeBaseChunkingConfigAdvanced;

export type KnowledgeGraphDocumentChunk = z.infer<typeof knowledgeGraphDocumentChunkSchema>;

const knowledgeBaseChunkingSentenceSplitterConfigSchema = z.object({
  chunk_size: z.number().int().min(1),
  chunk_overlap: z.number().int().min(0),
  paragraph_separator: z.string(),
}) satisfies z.ZodType<KnowledgeBaseChunkingSentenceSplitterConfig, any, any>;

const knowledgeBaseChunkingMarkdownSplitterConfigSchema = z.object({
  chunk_size: z.number().int().min(1),
  chunk_header_level: z.number().int().min(1).max(6),
}) satisfies z.ZodType<KnowledgeBaseChunkingMarkdownSplitterConfig, any, any>;

const knowledgeBaseChunkingSplitterRuleSchema = z.discriminatedUnion('splitter', [
  z.object({
    splitter: z.literal('MarkdownSplitter'),
    splitter_config: knowledgeBaseChunkingMarkdownSplitterConfigSchema,
  }),
  z.object({
    splitter: z.literal('SentenceSplitter'),
    splitter_config: knowledgeBaseChunkingSentenceSplitterConfigSchema,
  }),
]) satisfies z.ZodType<KnowledgeBaseChunkingSplitterRule, any, any>;

export const knowledgeBaseChunkingConfigSchema = z.discriminatedUnion('mode', [
  z.object({
    mode: z.literal('general'),
    chunk_size: z.number().int().min(1),
    chunk_overlap: z.number().int().min(0),
    paragraph_separator: z.string(),
  }),
  z.object({
    mode: z.literal('advanced'),
    rules: z.object({
      'text/plain': knowledgeBaseChunkingSplitterRuleSchema,
      'text/markdown': knowledgeBaseChunkingSplitterRuleSchema,
    }),
  }),
]) satisfies z.ZodType<KnowledgeBaseChunkingConfig, any, any>;

const knowledgeBaseSummarySchema = z.object({
  id: z.number(),
  name: z.string(),
  description: z.string().nullable(),
  index_methods: z.enum(['vector', 'knowledge_graph']).array(),
  documents_total: z.number().optional(),
  data_sources_total: z.number().optional(),
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
  creator: z.object({
    id: z.string(),
  }),
}) satisfies ZodType<KnowledgeBaseSummary, any, any>;

const knowledgeBaseSchema = knowledgeBaseSummarySchema.extend({
  data_sources: datasourceSchema.array(),
  llm: llmSummarySchema.nullable().optional(),
  embedding_model: embeddingModelSummarySchema.nullable().optional(),
  chunking_config: knowledgeBaseChunkingConfigSchema.nullable(),
}) satisfies ZodType<KnowledgeBase, any, any>;

const knowledgeGraphIndexProgressSchema = z.object({
  vector_index: indexSchema,
  documents: totalSchema,
  chunks: totalSchema,
  kg_index: indexSchema.optional(),
  entities: totalSchema.optional(),
  relationships: totalSchema.optional(),
}) satisfies ZodType<KnowledgeGraphIndexProgress>;

const knowledgeGraphDocumentChunkSchema = z.object({
  id: z.string(),
  document_id: z.number(),
  hash: z.string(),
  text: z.string(),
  meta: z.object({}).passthrough(),
  embedding: z.number().array(),
  relations: z.any(),
  source_uri: z.string(),
  index_status: indexStatusSchema,
  index_result: z.string().nullable(),
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
});

const vectorIndexErrorSchema = z.object({
  document_id: z.number(),
  document_name: z.string(),
  source_uri: z.string(),
  error: z.string().nullable(),
}) satisfies ZodType<DatasourceVectorIndexError, any, any>;

const kgIndexErrorSchema = z.object({
  document_id: z.number(),
  document_name: z.string(),
  chunk_id: z.string(),
  source_uri: z.string(),
  error: z.string().nullable(),
}) satisfies ZodType<DatasourceKgIndexError, any, any>;

const knowledgeBaseLinkedChatEngine = z.object({
  id: z.number(),
  name: z.string(),
  is_default: z.boolean(),
});

export async function listKnowledgeBases ({ page = 1, size = 10 }: PageParams) {
  return await fetch(requestUrl('/api/v1/admin/knowledge_bases', { page, size }), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(knowledgeBaseSummarySchema)));
}

export async function getKnowledgeBaseById (id: number): Promise<KnowledgeBase> {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(knowledgeBaseSchema));
}

export async function getKnowledgeBaseDocumentChunks (id: number, documentId: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}/documents/${documentId}/chunks`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(knowledgeGraphDocumentChunkSchema.array()));
}

export async function getKnowledgeBaseDocument (id: number, documentId: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}/documents/${documentId}`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(documentSchema.omit({ knowledge_base: true, data_source: true })));
}

export async function getKnowledgeBaseLinkedChatEngines (id: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}/linked_chat_engines`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(knowledgeBaseLinkedChatEngine.array()));
}

export async function deleteKnowledgeBaseDocument (id: number, documentId: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}/documents/${documentId}`), {
    method: 'DELETE',
    headers: await authenticationHeaders(),
  })
    .then(handleErrors);
}

export async function rebuildKBDocumentIndex (kb_id: number, doc_id: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${kb_id}/documents/${doc_id}/reindex`), {
    method: 'POST',
    headers: await authenticationHeaders(),
  })
    .then(handleErrors);
}

export async function createKnowledgeBase (params: CreateKnowledgeBaseParams) {
  return await fetch(requestUrl('/api/v1/admin/knowledge_bases'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...await authenticationHeaders(),
    },
    body: JSON.stringify(params),
  }).then(handleResponse(knowledgeBaseSchema));
}

export async function updateKnowledgeBase (id: number, params: UpdateKnowledgeBaseParams) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}`), {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...await authenticationHeaders(),
    },
    body: JSON.stringify(params),
  }).then(handleResponse(knowledgeBaseSchema));
}

export async function getKnowledgeGraphIndexProgress (id: number): Promise<KnowledgeGraphIndexProgress> {
  return fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}/overview`), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(knowledgeGraphIndexProgressSchema));
}

export async function listKnowledgeBaseVectorIndexErrors (id: number, { page = 1, size = 10 }: PageParams = {}) {
  return fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}/vector-index-errors`, { page, size }), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(zodPage(vectorIndexErrorSchema)));
}

export async function listKnowledgeBaseKgIndexErrors (id: number, { page = 1, size = 10 }: PageParams = {}) {
  return fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}/kg-index-errors`, { page, size }), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(zodPage(kgIndexErrorSchema)));
}

export async function retryKnowledgeBaseAllFailedTasks (id: number) {
  return fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}/retry-failed-index-tasks`), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
  }).then(handleErrors);
}

export async function deleteKnowledgeBase (id: number) {
  return await fetch(requestUrl(`/api/v1/admin/knowledge_bases/${id}`), {
    method: 'DELETE',
    headers: await authenticationHeaders(),
  })
    .then(handleErrors);
}
