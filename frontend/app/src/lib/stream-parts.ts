/**
 * Parser for the Vercel AI SDK "data stream protocol" (`<code>:<json>` lines) used by the backend chat stream API.
 * See `ChatEventType` in `backend/app/rag/types.py`.
 *
 * Ported from `parseStreamPart` of `ai@3.4` (Apache-2.0, https://github.com/vercel/ai) so the frontend no longer
 * depends on the `ai` package: newer `ai` majors dropped this protocol and the old versions have security advisories.
 */

export type JSONValue = null | string | number | boolean | { [key: string]: JSONValue } | JSONValue[];

export interface StreamPart<CODE extends string, NAME extends string, TYPE> {
  code: CODE;
  name: NAME;
  parse: (value: JSONValue) => { type: NAME; value: TYPE };
}

type Usage = { promptTokens: number; completionTokens: number };

const textStreamPart: StreamPart<'0', 'text', string> = {
  code: '0',
  name: 'text',
  parse: (value) => {
    if (typeof value !== 'string') {
      throw new Error('"text" parts expect a string value.');
    }
    return { type: 'text', value };
  },
};

const dataStreamPart: StreamPart<'2', 'data', JSONValue[]> = {
  code: '2',
  name: 'data',
  parse: (value) => {
    if (!Array.isArray(value)) {
      throw new Error('"data" parts expect an array value.');
    }
    return { type: 'data', value };
  },
};

const errorStreamPart: StreamPart<'3', 'error', string> = {
  code: '3',
  name: 'error',
  parse: (value) => {
    if (typeof value !== 'string') {
      throw new Error('"error" parts expect a string value.');
    }
    return { type: 'error', value };
  },
};

const messageAnnotationsStreamPart: StreamPart<'8', 'message_annotations', JSONValue[]> = {
  code: '8',
  name: 'message_annotations',
  parse: (value) => {
    if (!Array.isArray(value)) {
      throw new Error('"message_annotations" parts expect an array value.');
    }
    return { type: 'message_annotations', value };
  },
};

const toolCallStreamPart: StreamPart<'9', 'tool_call', { toolCallId: string; toolName: string; args: any }> = {
  code: '9',
  name: 'tool_call',
  parse: (value) => {
    if (value == null || typeof value !== 'object' || !('toolCallId' in value) || typeof value.toolCallId !== 'string' || !('toolName' in value) || typeof value.toolName !== 'string' || !('args' in value) || typeof value.args !== 'object') {
      throw new Error('"tool_call" parts expect an object with a "toolCallId", "toolName", and "args" property.');
    }
    return { type: 'tool_call', value: value as { toolCallId: string; toolName: string; args: any } };
  },
};

const toolResultStreamPart: StreamPart<'a', 'tool_result', { toolCallId: string; result: any }> = {
  code: 'a',
  name: 'tool_result',
  parse: (value) => {
    if (value == null || typeof value !== 'object' || !('toolCallId' in value) || typeof value.toolCallId !== 'string' || !('result' in value)) {
      throw new Error('"tool_result" parts expect an object with a "toolCallId" and a "result" property.');
    }
    return { type: 'tool_result', value: value as { toolCallId: string; result: any } };
  },
};

function parseUsage (value: { [key: string]: JSONValue }): Usage | undefined {
  const usage = value.usage;
  if (usage != null && typeof usage === 'object' && 'promptTokens' in usage && 'completionTokens' in usage) {
    return {
      promptTokens: typeof usage.promptTokens === 'number' ? usage.promptTokens : Number.NaN,
      completionTokens: typeof usage.completionTokens === 'number' ? usage.completionTokens : Number.NaN,
    };
  }
  return undefined;
}

const finishMessageStreamPart: StreamPart<'d', 'finish_message', { finishReason: string; usage?: Usage }> = {
  code: 'd',
  name: 'finish_message',
  parse: (value) => {
    if (value == null || typeof value !== 'object' || Array.isArray(value) || typeof value.finishReason !== 'string') {
      throw new Error('"finish_message" parts expect an object with a "finishReason" property.');
    }
    const result: { finishReason: string; usage?: Usage } = { finishReason: value.finishReason };
    const usage = parseUsage(value);
    if (usage) {
      result.usage = usage;
    }
    return { type: 'finish_message', value: result };
  },
};

const finishStepStreamPart: StreamPart<'e', 'finish_step', { finishReason: string; usage?: Usage; isContinued: boolean }> = {
  code: 'e',
  name: 'finish_step',
  parse: (value) => {
    if (value == null || typeof value !== 'object' || Array.isArray(value) || typeof value.finishReason !== 'string') {
      throw new Error('"finish_step" parts expect an object with a "finishReason" property.');
    }
    const result: { finishReason: string; usage?: Usage; isContinued: boolean } = {
      finishReason: value.finishReason,
      isContinued: typeof value.isContinued === 'boolean' ? value.isContinued : false,
    };
    const usage = parseUsage(value);
    if (usage) {
      result.usage = usage;
    }
    return { type: 'finish_step', value: result };
  },
};

/**
 * Parts of the protocol that the chat UI does not handle. They are passed through without validation,
 * the chat controller ignores them with a warning.
 */
function unhandledStreamPart<CODE extends string, NAME extends string> (code: CODE, name: NAME): StreamPart<CODE, NAME, JSONValue> {
  return { code, name, parse: (value) => ({ type: name, value }) };
}

const streamParts = [
  textStreamPart,
  unhandledStreamPart('1', 'function_call'),
  dataStreamPart,
  errorStreamPart,
  unhandledStreamPart('4', 'assistant_message'),
  unhandledStreamPart('5', 'assistant_control_data'),
  unhandledStreamPart('6', 'data_message'),
  unhandledStreamPart('7', 'tool_calls'),
  messageAnnotationsStreamPart,
  toolCallStreamPart,
  toolResultStreamPart,
  unhandledStreamPart('b', 'tool_call_streaming_start'),
  unhandledStreamPart('c', 'tool_call_delta'),
  finishMessageStreamPart,
  finishStepStreamPart,
] as const;

export type StreamPartType = ReturnType<(typeof streamParts)[number]['parse']>;

const streamPartsByCode = new Map<string, (typeof streamParts)[number]>(streamParts.map(part => [part.code, part]));

export function parseStreamPart (line: string): StreamPartType {
  const firstSeparatorIndex = line.indexOf(':');
  if (firstSeparatorIndex === -1) {
    throw new Error('Failed to parse stream string. No separator found.');
  }

  const prefix = line.slice(0, firstSeparatorIndex);
  const streamPart = streamPartsByCode.get(prefix);
  if (!streamPart) {
    throw new Error(`Failed to parse stream string. Invalid code ${prefix}.`);
  }

  const jsonValue: JSONValue = JSON.parse(line.slice(firstSeparatorIndex + 1));
  return streamPart.parse(jsonValue);
}
