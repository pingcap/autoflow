import { chat, type Chat, type ChatMessage, type PostChatParams } from '@/api/chats';
import { BaseChatMessageController, ChatMessageController, LegacyChatMessageController, type OngoingState, StackVMChatMessageController } from '@/components/chat/chat-message-controller';
import { AppChatStreamState, type BaseAnnotation, chatDataPartSchema, fixChatInitialData, type StackVMState } from '@/components/chat/chat-stream-state';
import type { GtagFn } from '@/components/gtag-provider';
import { getErrorMessage } from '@/lib/errors';
import { trigger } from '@/lib/react';
import { type JSONValue, type StreamPart } from 'ai';
import EventEmitter from 'eventemitter3';

export interface ChatControllerEventsMap<State = AppChatStreamState, Annotation extends BaseAnnotation<State> = BaseAnnotation<State>> {
  'created': [Chat];
  'updated': [Chat];
  'message-loaded': [messageController: BaseChatMessageController<State, Annotation>];

  /**
   * Emit instantly when {@link ChatController#post} is called
   */
  'post': [params: Omit<PostChatParams, 'chat_id'>];

  /**
   * Emit when server returned chat and chat_message info
   */
  'post-initialized': [];
  'post-finished': [];
  'post-error': [error: unknown];

  /**
   * Experimental
   */
  'ui:input-mount': [HTMLTextAreaElement | HTMLInputElement];
  'ui:input-unmount': [HTMLTextAreaElement | HTMLInputElement];
}

export class ChatController<State extends AppChatStreamState = AppChatStreamState, Annotation extends BaseAnnotation<State> = BaseAnnotation<State>> extends EventEmitter<ChatControllerEventsMap<State, Annotation>> {
  public chat: Chat | undefined;

  private _messages: Map<number, ChatMessageController | StackVMChatMessageController> = new Map();

  private _postParams: Omit<PostChatParams, 'chat_id'> | undefined = undefined;
  private _postError: unknown = undefined;
  private _postInitialized: boolean = false;

  private _inputElement: HTMLTextAreaElement | HTMLInputElement | null = null;

  get postState () {
    return {
      params: this._postParams,
      error: this._postError,
      initialized: this._postInitialized,
    };
  }

  constructor (
    chat: Chat | undefined = undefined,
    messages: ChatMessage[] | undefined = [],
    initialPost: Omit<PostChatParams, 'chat_id'> | undefined = undefined,
    inputElement: HTMLInputElement | HTMLTextAreaElement | null = null,
    private readonly _gtagFn: GtagFn = () => {},
  ) {
    super();
    if (chat) {
      this.chat = chat;
    }
    for (const message of messages) {
      this.upsertMessage(message);
    }
    if (initialPost) {
      this.post(initialPost);
    }
    this._inputElement = inputElement;
    if (inputElement) {
      this.emit('ui:input-mount', inputElement);
    }
  }

  get inputElement () {
    return this._inputElement;
  }

  set inputElement (value: HTMLInputElement | HTMLTextAreaElement | null) {
    if (this._inputElement) {
      if (value) {
        if (value !== this._inputElement) {
          const old = this._inputElement;
          this._inputElement = null;
          this.emit('ui:input-unmount', old);

          this._inputElement = value;
          this.emit('ui:input-mount', value);
        }
      } else {
        const old = this._inputElement;
        this._inputElement = null;
        this.emit('ui:input-unmount', old);
      }
    } else {
      if (value) {
        this._inputElement = value;
        this.emit('ui:input-mount', value);
      }
    }
  }

  private get _enabledInputElement () {
    if (!this._inputElement) {
      console.warn('Input element is not exists.');
      return;
    }
    if (this._inputElement.disabled) {
      console.warn('Input element is disabled currently.');
      return;
    }

    return this._inputElement;
  }

  get inputEnabled () {
    if (!this._inputElement) {
      return false;
    }

    return !this._inputElement.disabled;
  }

  get input (): string {
    return this._inputElement?.value ?? '';
  }

  set input (value: string) {
    const inputElement = this._enabledInputElement;
    if (inputElement) {
      trigger(inputElement as HTMLTextAreaElement, HTMLTextAreaElement, value);
    }
  }

  focusInput () {
    const inputElement = this._enabledInputElement;
    if (inputElement) {
      inputElement.focus();
    }
  }

  get messages (): (ChatMessageController | StackVMChatMessageController)[] {
    return Array.from(this._messages.values()).sort((a, b) => a.message.ordinal - b.message.ordinal);
  }

  async post (params: Omit<PostChatParams, 'chat_id'>) {
    if (this._postParams) {
      throw new Error('previous not finished.');
    }

    if (!params.content.trim()) {
      throw new Error('Empty message');
    }

    this._gtagFn('event', 'tidbai.events.message-start', {
      'tidbai_appending_message': !!this.chat?.id,
    });

    // Initialize post states
    this._postParams = params;
    this._postError = undefined;
    this._postInitialized = false;
    this.emit('post', params);

    let ongoingMessageController: ChatMessageController | StackVMChatMessageController | undefined = undefined;

    try {
      const streamParts = chat({
        chat_id: this.chat?.id,
        ...params,
      });

      // Process stream parts and dispatch to ongoingMessageController
      for await (let part of streamParts) {
        ongoingMessageController = this._processPart(ongoingMessageController, part);
      }

      // Cleanup post states
      if (ongoingMessageController) {
        this.upsertMessage(ongoingMessageController.finish());
      } else {
        console.warn('Empty ongoing message');
      }

      this._postParams = undefined;
      this._postInitialized = false;
      this.emit('post-finished');

      this._gtagFn('event', 'tidbai.events.message-finish', {});
    } catch (error) {
      ongoingMessageController?.applyError(getErrorMessage(error));
      this._postError = error;
      this.emit('post-error', error);

      this._gtagFn('event', 'tidbai.events.message-error', {});
    }
  }

  // TODO: wait server implementation
  async regenerate (messageId: number) {
    throw new Error('not supported.');
  }

  updateChat (chat: Chat): void {
    const newCreated = !this.chat;
    this.chat = { ...this.chat, ...chat };
    this.emit(newCreated ? 'created' : 'updated', this.chat);
  }

  upsertMessage (message: ChatMessage): void {
    let controller = this._messages.get(message.id);

    if (controller) {
      controller.update(message);
    } else {
      this.createMessage(message);
    }
  }

  _processPart (ongoingMessageController: ChatMessageController | StackVMChatMessageController | undefined, part: ReturnType<StreamPart<any, any, any>['parse']>) {
    switch (part.type) {
      case 'data':
        // Data part contains chat and chat_message info from server. will be sent twice (beginning and finished).
        // We will update frontend cached and computed info like message content which is computed from stream text deltas.
        ongoingMessageController = this._processDataPart(ongoingMessageController, part);
        break;
      case 'message_annotations':
        // Message annotations part containing current generating state.
        this._processMessageAnnotationPart(ongoingMessageController, part);
        break;
      case 'text':
        this._processTextPart(ongoingMessageController, part);
        break;
      case 'error':
        this._processErrorPart(ongoingMessageController, part);
        break;
      case 'tool_call':
        this._processToolCallPart(ongoingMessageController, part);
        break;
      case 'tool_result':
        this._processToolResultPart(ongoingMessageController, part);
        break
      case 'finish_step':

      case 'finish_message':
        // TODO
        break;
      default:
        console.warn('unsupported stream part', part);
    }
    return ongoingMessageController;
  }

  private _processDataPart (ongoingMessageController: ChatMessageController | StackVMChatMessageController | undefined, part: ReturnType<StreamPart<any, 'data', JSONValue[]>['parse']>): ChatMessageController | StackVMChatMessageController {
    const { chat, user_message, assistant_message } = chatDataPartSchema.parse(fixChatInitialData(part.value[0]));
    this.updateChat(chat);
    this.upsertMessage(user_message);
    if (!ongoingMessageController) {
      ongoingMessageController = this.createMessage(assistant_message, true);
      this._postInitialized = true;
      this.emit('post-initialized');
    } else {
      ongoingMessageController.update(assistant_message);
    }

    return ongoingMessageController;
  }

  private _processMessageAnnotationPart (ongoingMessageController: ChatMessageController | StackVMChatMessageController | undefined, part: ReturnType<StreamPart<any, 'message_annotations', JSONValue[]>['parse']>) {
    assertNonNull(ongoingMessageController, 'Cannot handle chat stream part: no ongoingMessageController', part);
    const annotation = ongoingMessageController.parseAnnotation(part.value[0]);
    ongoingMessageController.applyStreamAnnotation(annotation as never);
  }

  private _processTextPart (ongoingMessageController: ChatMessageController | StackVMChatMessageController | undefined, part: ReturnType<StreamPart<any, 'text', string>['parse']>) {
    if (part.value) { // ignore leading empty chunks.
      assertNonNull(ongoingMessageController, 'Cannot handle chat stream part: no ongoingMessageController', part);
      ongoingMessageController.applyDelta(part.value);
    }
  }

  private _processErrorPart (ongoingMessageController: ChatMessageController | StackVMChatMessageController | undefined, part: ReturnType<StreamPart<any, 'error', string>['parse']>) {
    assertNonNull(ongoingMessageController, 'Cannot handle chat stream part: no ongoingMessageController', part);
    ongoingMessageController.applyError(part.value);
  }

  private _processToolCallPart (ongoingMessageController: ChatMessageController | StackVMChatMessageController | undefined, part: ReturnType<StreamPart<any, 'tool_call', { toolCallId: string, toolName: string, args: any }>['parse']>) {
    assertNonNull(ongoingMessageController, 'Cannot handle chat stream part: no ongoingMessageController', part);
    ongoingMessageController.applyToolCall(part.value);
  }

  private _processToolResultPart (ongoingMessageController: ChatMessageController | StackVMChatMessageController | undefined, part: ReturnType<StreamPart<any, 'tool_call', { toolCallId: string, result: any }>['parse']>) {
    assertNonNull(ongoingMessageController, 'Cannot handle chat stream part: no ongoingMessageController', part);
    ongoingMessageController.applyToolResult(part.value);
  }

  private createMessage (message: ChatMessage, initialOngoingState?: true) {
    if (!this.chat?.engine_options) {
      throw new Error('Unable to decide which chat engine used.');
    }

    if (this.chat.engine_options.external_engine_config?.stream_chat_api_url) {
      return this.createStackVMMessage(message, initialOngoingState);
    } else {
      return this.createLegacyMessage(message, initialOngoingState);
    }
  }

  private createLegacyMessage (message: ChatMessage, initialOngoingState?: true | OngoingState) {
    const controller = new LegacyChatMessageController(message, initialOngoingState);
    this._messages.set(message.id, controller);
    this.emit('message-loaded', controller as any);
    return controller;
  }

  private createStackVMMessage (message: ChatMessage, initialOngoingState?: true | OngoingState<StackVMState>) {
    const controller = new StackVMChatMessageController(message, initialOngoingState);
    this._messages.set(message.id, controller);
    this.emit('message-loaded', controller as any);
    return controller;
  }
}

function assertNonNull<T> (value: T, message: string, ...args: any): asserts value is NonNullable<T> {
  if (value == null) {
    console.warn(message, args);
    throw new Error('bad stream');
  }
}
