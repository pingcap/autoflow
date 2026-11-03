import { parseStreamPart } from '@/lib/stream-parts';

test('parse stream parts', () => {
  expect(parseStreamPart('0:"Hello"')).toEqual({ type: 'text', value: 'Hello' });
  expect(parseStreamPart('2:[{"chat":{"id":"x"}}]')).toEqual({ type: 'data', value: [{ chat: { id: 'x' } }] });
  expect(parseStreamPart('3:"error"')).toEqual({ type: 'error', value: 'error' });
  expect(parseStreamPart('8:[{"state":1}]')).toEqual({ type: 'message_annotations', value: [{ state: 1 }] });
  expect(parseStreamPart('9:{"toolCallId":"1","toolName":"t","args":{}}')).toEqual({ type: 'tool_call', value: { toolCallId: '1', toolName: 't', args: {} } });
  expect(parseStreamPart('a:{"toolCallId":"1","result":null}')).toEqual({ type: 'tool_result', value: { toolCallId: '1', result: null } });
  expect(parseStreamPart('d:{"finishReason":"stop","usage":{"promptTokens":1,"completionTokens":2}}')).toEqual({ type: 'finish_message', value: { finishReason: 'stop', usage: { promptTokens: 1, completionTokens: 2 } } });
  expect(parseStreamPart('e:{"finishReason":"stop"}')).toEqual({ type: 'finish_step', value: { finishReason: 'stop', isContinued: false } });
});

test('invalid stream parts', () => {
  expect(() => parseStreamPart('no separator')).toThrow('No separator found');
  expect(() => parseStreamPart('z:"x"')).toThrow('Invalid code z');
  expect(() => parseStreamPart('0:1')).toThrow('"text" parts expect a string value.');
  expect(() => parseStreamPart('2:{}')).toThrow('"data" parts expect an array value.');
});
