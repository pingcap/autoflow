import { ChatMessageRole } from '@/api/chats';
import type { ChatMessageController } from '@/components/chat/chat-message-controller';
import type { ChatInitialData } from '@/components/chat/chat-stream-state';
import { jest } from '@jest/globals';
import { z } from 'zod';

jest.unstable_mockModule('../../api/chats', () => ({
  chatMessageSchema: z.any(),
  chatSchema: z.any(),
  chat: (...args: any) => currentChat(...args),
}));

let currentChat: any;

afterAll(() => {
  // using pnpm patch https://github.com/jestjs/jest/pull/15080/files#diff-c0d5b59e96fdc7ffc98405e8afb46d525505bc7b1c24916b5c8482de5a186c00
  jest.unstable_unmockModule('../../api/chats');
});

const exampleData = {
  chat: {
    id: 'mock',
    created_at: new Date,
    updated_at: new Date,
    browser_id: null,
    deleted_at: null,
    engine_options: {
      llm: {
        condense_question_prompt: '',
        text_qa_prompt: '',
        refine_prompt: '',
        intent_graph_knowledge: '',
        normal_graph_knowledge: '',
      },
      knowledge_graph: {
        depth: 0,
        enabled: false,
        include_meta: false,
        with_degree: false,
        using_intent_search: false,
      },
    },
    user_id: '0',
    engine_id: 1,
    title: 'Demo',
  },
  assistant_message: {
    chat_id: 'mock',
    content: '',
    created_at: new Date(),
    error: null,
    finished_at: new Date(),
    id: 2,
    ordinal: 1,
    role: ChatMessageRole.assistant,
    sources: [],
    trace_url: '',
    updated_at: new Date(),
    user_id: '0',

  },
  user_message: {
    chat_id: 'mock',
    content: 'ping',
    created_at: new Date(),
    error: null,
    finished_at: new Date(),
    id: 1,
    ordinal: 0,
    role: ChatMessageRole.user,
    sources: [],
    trace_url: '',
    updated_at: new Date(),
    user_id: '0',
  },
} satisfies ChatInitialData;

describe('chat protocol', () => {
  test('terminate before server responses', async () => {
    const error = new Error('terminate before server response');
    currentChat = () => {
      throw error;
    };

    // for using `jest.unstable_mockModule` mocked module
    const { ChatController } = await import('./chat-controller');

    const controller = new ChatController();
    const onPost = jest.fn();
    const onPostInitialized = jest.fn();
    const onPostError = jest.fn();

    controller
      .on('post', onPost)
      .on('post-initialized', onPostInitialized)
      .on('post-error', onPostError);

    await controller.post({ content: 'hi' }).catch(() => {});
    expect(onPost).toHaveBeenCalledTimes(1);
    expect(onPost).toHaveBeenCalledWith({ content: 'hi' });

    expect(onPostInitialized).toHaveBeenCalledTimes(0);

    expect(onPostError).toHaveBeenCalledTimes(1);
    expect(onPostError).toHaveBeenCalledWith(error);
  });

  test('terminated while generating', async () => {
    const error = new Error('ee');
    currentChat = async function* () {
      yield {
        type: 'data',
        value: [exampleData],
      };
      yield { type: 'text', value: 'pong' };
      yield { type: 'error', value: 'terminated' };
    };

    // for using `jest.unstable_mockModule` mocked module
    const { ChatController } = await import('./chat-controller');

    const controller = new ChatController();
    const onPost = jest.fn();
    const onPostInitialized = jest.fn();
    const onMessageLoaded = jest.fn();
    const onPostError = jest.fn();

    controller
      .on('post', onPost)
      .on('post-initialized', onPostInitialized)
      .on('post-error', onPostError)
      .on('message-loaded', onMessageLoaded);

    await controller.post({ content: 'hi' }).catch(() => {});

    expect(onPost).toHaveBeenCalledTimes(1);
    expect(onPost).toHaveBeenCalledWith({ content: 'hi' });

    expect(onPostInitialized).toHaveBeenCalledTimes(1);
    expect(onPostError).toHaveBeenCalledTimes(0);

    expect(onMessageLoaded).toHaveBeenCalledTimes(2);

    const assistantMessage: ChatMessageController = onMessageLoaded.mock.calls.find(msg => (msg[0] as ChatMessageController).role === 'assistant')![0] as any;
    expect(assistantMessage.content).toBe('pong');
    expect(assistantMessage.message.error).toBe('terminated');
  });

  test('ok', async () => {
    currentChat = async function* () {
      yield {
        type: 'data',
        value: [exampleData],
      };
      yield { type: 'text', value: 'pong' };
    };

    const { ChatController } = await import('./chat-controller');

    const controller = new ChatController();

    const onPostInitialized = jest.fn();
    const onMessageLoaded = jest.fn();
    const onPostFinished = jest.fn();

    controller
      .on('post-initialized', onPostInitialized)
      .on('message-loaded', onMessageLoaded)
      .on('post-finished', onPostFinished);

    await controller.post({ content: 'ping' }).catch((err) => {console.error(err);});
    expect(onPostInitialized).toHaveBeenCalledTimes(1);
    expect(onMessageLoaded).toHaveBeenCalledTimes(2);
    expect(onPostFinished).toHaveBeenCalledTimes(1);

    const assistantMessage: ChatMessageController = onMessageLoaded.mock.calls.find(msg => (msg[0] as ChatMessageController).role === 'assistant')![0] as any;
    expect(assistantMessage.content).toBe('pong');
  });
});
