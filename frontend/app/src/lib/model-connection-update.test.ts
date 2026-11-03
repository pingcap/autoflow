import { modelConnectionUpdate } from './model-connection-update';

test('endpoint and fresh credentials are sent in one update', () => {
  const result = modelConnectionUpdate('{"api_base":"https://provider.example/v1"}', 'fixture-key', 'str');
  expect(result).toEqual({ config: { api_base: 'https://provider.example/v1' }, credentials: 'fixture-key' });
});

test('JSON credentials never become the unencrypted configuration', () => {
  const result = modelConnectionUpdate('{"location":"us-west1"}', '{"private_key":"fixture-only"}', 'dict');
  expect(result.credentials).toEqual({ private_key: 'fixture-only' });
  expect(result.config).toEqual({ location: 'us-west1' });
});

test('blank credentials are omitted and invalid JSON errors contain no secret', () => {
  expect(modelConnectionUpdate('{}', '', 'str')).toEqual({ config: {} });
  expect(() => modelConnectionUpdate('{}', 'fixture-private-key', 'dict')).toThrow('Credentials must be a valid JSON object.');
});
