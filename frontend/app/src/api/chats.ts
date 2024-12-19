import type { ChatEngineOptions } from '@/api/chat-engines';
import { type KnowledgeGraph, knowledgeGraphSchema } from '@/api/graph';
import { bufferedReadableStreamTransformer } from '@/lib/buffered-readable-stream';
import { authenticationHeaders, handleErrors, handleResponse, type Page, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { parseStreamPart } from 'ai';
import { z, type ZodType } from 'zod';

type ClientEngineOptions = Omit<ChatEngineOptions, 'post_verification_token'>;

export interface Chat {
  title: string;
  engine_id: number;
  engine_options: ClientEngineOptions;
  deleted_at: Date | null;
  user_id: string | null;
  browser_id: string | null;
  updated_at: Date;
  created_at: Date;
  id: string;
  origin: string | null;
}

export interface ChatDetail {
  chat: Chat;
  messages: ChatMessage[];
}

export const enum ChatMessageRole {
  assistant = 'assistant',
  user = 'user'
}

export interface ChatMessage {
  id: number;
  role: ChatMessageRole;
  error: string | null;
  trace_url: string | null;
  finished_at: Date | null;
  user_id: string | null;
  created_at: Date;
  updated_at: Date;
  ordinal: number;
  content: string;
  sources: ChatMessageSource[];
  chat_id: string;
  post_verification_result_url: string | null;
}

export interface ChatMessageSource {
  id: number;
  name: string;
  source_uri: string;
}

export const chatSchema = z.object({
  title: z.string(),
  engine_id: z.number(),
  engine_options: z.object({}).passthrough().transform(value => value as never as ChatEngineOptions),
  deleted_at: zodJsonDate().nullable(),
  user_id: z.string().nullable(),
  browser_id: z.string().nullable(),
  updated_at: zodJsonDate(),
  created_at: zodJsonDate(),
  id: z.string(),
  origin: z.string().nullable(),
}) satisfies ZodType<Chat, any, any>;

const chatMessageSourceSchema = z.object({
  id: z.number(),
  name: z.string(),
  source_uri: z.string(),
});

export const chatMessageSchema = z.object({
  id: z.number(),
  role: z.enum([ChatMessageRole.user, ChatMessageRole.assistant]),
  error: z.string().nullable(),
  trace_url: z.string().nullable(),
  finished_at: zodJsonDate().nullable(),
  user_id: z.string().nullable(),
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
  ordinal: z.number(),
  content: z.string(),
  sources: chatMessageSourceSchema.array(),
  chat_id: z.string(),
  post_verification_result_url: z.string().url().nullable(),
}) satisfies ZodType<ChatMessage, any, any>;

const chatDetailSchema = z.object({
  chat: chatSchema,
  messages: chatMessageSchema.array(),
});

export interface FeedbackParams {
  feedback_type: 'like' | 'dislike';
  comment: string;
}

export interface PostChatParams {
  chat_id?: string;
  chat_engine?: string;
  content: string;

  headers?: HeadersInit;
  signal?: AbortSignal;
}

export async function listChats ({ page = 1, size = 10 }: PageParams = {}): Promise<Page<Chat>> {
  return await fetch(requestUrl('/api/v1/chats', { page, size }), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(chatSchema)));
}

export async function getChat (id: string): Promise<ChatDetail> {
  return await fetch(requestUrl(`/api/v1/chats/${id}`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(chatDetailSchema));
}

export async function deleteChat (id: string): Promise<void> {
  await fetch(requestUrl(`/api/v1/chats/${id}`), {
    method: 'delete',
    headers: await authenticationHeaders(),
  }).then(handleErrors);
}

export async function postFeedback (chatMessageId: number, feedback: FeedbackParams) {
  return await fetch(requestUrl(`/api/v1/chat-messages/${chatMessageId}/feedback`), {
    method: 'post',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify(feedback),
  }).then(handleErrors);
}

export async function getChatMessageSubgraph (chatMessageId: number): Promise<KnowledgeGraph> {
  return await fetch(requestUrl(`/api/v1/chat-messages/${chatMessageId}/subgraph`), {
    headers: await authenticationHeaders(),
    credentials: 'include',
  })
    .then(handleResponse(knowledgeGraphSchema));
}

export async function getChatMessageRecommendedQuestions (chatMessageId: number) {
  return await fetch(requestUrl(`/api/v1/chat-messages/${chatMessageId}/recommended-questions`), {
    headers: await authenticationHeaders(),
    credentials: 'include',
  })
    .then(handleResponse(z.string().array()));
}

export async function reloadChatMessageRecommendedQuestions (chatMessageId: number) {
  return await fetch(requestUrl(`/api/v1/chat-messages/${chatMessageId}/recommended-questions`), {
    method: 'POST',
    headers: await authenticationHeaders(),
    credentials: 'include',
  })
    .then(handleResponse(z.string().array()));
}

export async function* chat ({ chat_id, chat_engine, content, headers: headersInit, signal }: PostChatParams, onResponse?: (response: Response) => void) {
  const headers = new Headers(headersInit);
  headers.set('Content-Type', 'application/json');

  for (let [key, value] of Object.entries(await authenticationHeaders())) {
    headers.set(key, value);
  }

  const response = await fetch(requestUrl(`/api/v1/chats`), {
    method: 'POST',
    headers,
    credentials: 'include',
    body: JSON.stringify({
      chat_id,
      chat_engine,
      stream: true,
      messages: [{
        'role': 'user',
        content,
      }],
    }),
    signal,
  }).then(handleErrors);

  onResponse?.(response);

  if (!response.body) {
    throw new Error(`${response.status} ${response.statusText} Empty response body`);
  }

  const reader = response.body.pipeThrough(bufferedReadableStreamTransformer()).getReader();

  while (true) {
    const chunk = await reader.read();
    if (chunk.done) {
      break;
    }

    if (!!chunk.value.trim()) {
      yield parseStreamPart(chunk.value);
    }
  }
}
