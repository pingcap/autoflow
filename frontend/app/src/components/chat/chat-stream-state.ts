/**
 *  TRACE = 0
 *     SOURCE_NODES = 1
 *     KG_RETRIEVAL = 2
 *     REFINE_QUESTION = 3
 *     SEARCH_RELATED_DOCUMENTS = 4
 *     GENERATE_ANSWER = 5
 *     FINISHED = 9
 */
import { Chat, ChatMessage, chatMessageSchema, ChatMessageSource, chatSchema } from '@/api/chats';
import { StackVM } from '@/lib/stackvm';
import { z, type ZodType } from 'zod';

export const enum BaseState {
  CONNECTING = 'CONNECTING', // only client side
  UNKNOWN = 'UNKNOWN',
}

export const enum AppChatStreamState {
  CONNECTING = 'CONNECTING', // only client side
  TRACE = 'TRACE',
  SOURCE_NODES = 'SOURCE_NODES',
  KG_RETRIEVAL = 'KG_RETRIEVAL',
  REFINE_QUESTION = 'REFINE_QUESTION',
  SEARCH_RELATED_DOCUMENTS = 'SEARCH_RELATED_DOCUMENTS',
  RERANKING = 'RERANKING',
  GENERATE_ANSWER = 'GENERATE_ANSWER',
  FINISHED = 'FINISHED',
  FAILED = 'FAILED',
  UNKNOWN = 'UNKNOWN',
}

export type StackVMState = {
  task_id: string;
  branch: string;
  state: StackVM.model.ParsedState;
  seq_no: number;
  toolCalls: StackVMToolCall[];
};

export type StackVMToolCall = { toolCallId: string, toolName: string, args: any, result?: any }

export interface BaseAnnotation<S = AppChatStreamState> {
  state: S;
  display?: string;
}

export interface TraceAnnotation extends BaseAnnotation<AppChatStreamState.TRACE> {
  context: { langfuse_url: string };
}

export interface SourceNodesAnnotation extends BaseAnnotation<AppChatStreamState.SOURCE_NODES> {
  context: ChatMessageSource[];
}

export interface RefineQuestionAnnotation extends BaseAnnotation<AppChatStreamState.REFINE_QUESTION> {
  message?: string;
}

export type ChatMessageAnnotation =
  BaseAnnotation<Exclude<AppChatStreamState, AppChatStreamState.TRACE | AppChatStreamState.SOURCE_NODES | AppChatStreamState.REFINE_QUESTION>>
  | TraceAnnotation
  | SourceNodesAnnotation
  | RefineQuestionAnnotation;

export interface StackVMStateAnnotation extends BaseAnnotation<StackVMState> {
}

export type ChatInitialData = {
  chat: Chat;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

// FIXME: Server will not return updated_at
export function fixChatInitialData (data: any) {
  if (data.assistant_message) {
    if (!data.assistant_message.updated_at) {
      data.assistant_message.updated_at = data.assistant_message.created_at;
    }
  }
  return data as any;
}

export const chatDataPartSchema = z.object({
  chat: chatSchema,
  user_message: chatMessageSchema,
  assistant_message: chatMessageSchema,
}) satisfies ZodType<ChatInitialData, any>;
