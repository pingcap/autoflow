import { authenticationHeaders, handleErrors, handleResponse, type Page, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { number, z, type ZodType } from 'zod';

export interface ChatEngine {
  id: number;
  name: string;
  updated_at: Date;
  created_at: Date;
  deleted_at: Date | null;
  engine_options: ChatEngineOptions;
  llm_id: number | null;
  fast_llm_id: number | null;
  reranker_id: number | null;
  is_default: boolean;
}

export interface CreateChatEngineParams {
  name: string;
  engine_options: ChatEngineOptions;
  llm_id?: number | null;
  fast_llm_id?: number | null;
  reranker_id?: number | null;
}

export interface ChatEngineOptions {
  external_engine_config?: {
    stream_chat_api_url?: string | null
  } | null;
  clarify_question?: boolean | null;
  knowledge_base?: ChatEngineKnowledgeBaseOptions | null;
  knowledge_graph?: ChatEngineKnowledgeGraphOptions | null;
  llm?: ChatEngineLLMOptions | null;
  post_verification_url?: string | null;
  post_verification_token?: string | null;
  hide_sources?: boolean | null;
}

export interface ChatEngineKnowledgeBaseOptions {
  linked_knowledge_base?: LinkedKnowledgeBaseOptions | null;
}

export interface ChatEngineKnowledgeGraphOptions {
  depth?: number | null;
  enabled?: boolean | null;
  include_meta?: boolean | null;
  with_degree?: boolean | null;
  using_intent_search?: boolean | null;
}

export type ChatEngineLLMOptions = {
  condense_question_prompt?: string | null
  condense_answer_prompt?: string | null
  text_qa_prompt?: string | null
  refine_prompt?: string | null
  intent_graph_knowledge?: string | null
  normal_graph_knowledge?: string | null
  clarifying_question_prompt?: string | null
  generate_goal_prompt?: string | null
  further_questions_prompt?: string | null
}

export interface LinkedKnowledgeBaseOptions {
  id?: number | null;
}

const kbOptionsSchema = z.object({
  linked_knowledge_base: z.object({ id: number().nullable().optional() }).nullable().optional(),
}).passthrough();

const kgOptionsSchema = z.object({
  depth: z.number().nullable().optional(),
  enabled: z.boolean().nullable().optional(),
  include_meta: z.boolean().nullable().optional(),
  with_degree: z.boolean().nullable().optional(),
  using_intent_search: z.boolean().nullable().optional(),
}).passthrough() satisfies ZodType<ChatEngineKnowledgeGraphOptions>;

const llmOptionsSchema =
  z.object({
    condense_question_prompt: z.string().nullable().optional(),
    condense_answer_prompt: z.string().nullable().optional(),
    text_qa_prompt: z.string().nullable().optional(),
    refine_prompt: z.string().nullable().optional(),
    intent_graph_knowledge: z.string().nullable().optional(),
    normal_graph_knowledge: z.string().nullable().optional(),
    clarifying_question_prompt: z.string().nullable().optional(),
    generate_goal_prompt: z.string().nullable().optional(),
    further_questions_prompt: z.string().nullable().optional(),
    // provider: z.string(),
    // reranker_provider: z.string(),
    // reranker_top_k: z.number(),
  }).passthrough() as ZodType<ChatEngineLLMOptions, any, any>;

const chatEngineOptionsSchema = z.object({
  external_engine_config: z.object({
    stream_chat_api_url: z.string().optional().nullable(),
  }).nullable().optional(),
  clarify_question: z.boolean().nullable().optional(),
  knowledge_base: kbOptionsSchema.nullable().optional(),
  knowledge_graph: kgOptionsSchema.nullable().optional(),
  llm: llmOptionsSchema.nullable().optional(),
  post_verification_url: z.string().nullable().optional(),
  post_verification_token: z.string().nullable().optional(),
  hide_sources: z.boolean().nullable().optional(),
}).passthrough() satisfies ZodType<ChatEngineOptions, any, any>;

const chatEngineSchema = z.object({
  id: z.number(),
  name: z.string(),
  updated_at: zodJsonDate(),
  created_at: zodJsonDate(),
  deleted_at: zodJsonDate().nullable(),
  engine_options: chatEngineOptionsSchema,
  llm_id: z.number().nullable(),
  fast_llm_id: z.number().nullable(),
  reranker_id: z.number().nullable(),
  is_default: z.boolean(),
}) satisfies ZodType<ChatEngine, any, any>;

export async function getDefaultChatEngineOptions (): Promise<ChatEngineOptions> {
  return await fetch(requestUrl('/api/v1/admin/chat-engines-default-config'), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(chatEngineOptionsSchema));
}

export async function listChatEngines ({ page = 1, size = 10 }: PageParams = {}): Promise<Page<ChatEngine>> {
  return await fetch(requestUrl('/api/v1/admin/chat-engines', { page, size }), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(chatEngineSchema)));
}

export async function getChatEngine (id: number): Promise<ChatEngine> {
  return await fetch(requestUrl(`/api/v1/admin/chat-engines/${id}`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(chatEngineSchema));
}

export async function updateChatEngine (id: number, partial: Partial<Pick<ChatEngine, 'name' | 'llm_id' | 'fast_llm_id' | 'reranker_id' | 'engine_options' | 'is_default'>>): Promise<void> {
  await fetch(requestUrl(`/api/v1/admin/chat-engines/${id}`), {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...await authenticationHeaders(),
    },
    body: JSON.stringify(partial),
  })
    .then(handleErrors);
}

export async function createChatEngine (create: CreateChatEngineParams) {
  return await fetch(requestUrl(`/api/v1/admin/chat-engines`), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...await authenticationHeaders(),
    },
    body: JSON.stringify(create),
  })
    .then(handleResponse(chatEngineSchema));
}

export async function deleteChatEngine (id: number): Promise<void> {
  await fetch(requestUrl(`/api/v1/admin/chat-engines/${id}`), {
    method: 'DELETE',
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleErrors);
}
