import { jest } from '@jest/globals';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { LangfuseConfiguration, LangfuseUpdate } from '@/api/langfuse';

const original: LangfuseConfiguration = {
  host: 'https://us.cloud.langfuse.com', public_key: 'pk-fixture',
  revision: 'revision-one', configured: true, migration_required: false,
};
const update = jest.fn<(value: LangfuseUpdate) => Promise<LangfuseConfiguration>>();
jest.unstable_mockModule('../../api/langfuse', () => ({
  getLangfuseConfiguration: jest.fn(),
  updateLangfuseConfiguration: update,
}));
const { LangfuseConfigurationForm } = await import('./IntegrationsSettings');

afterEach(() => { cleanup(); jest.clearAllMocks(); });

test('saved secrets stay absent and replacement submits one destination-bound update', async () => {
  const saved = jest.fn<(value: LangfuseConfiguration) => Promise<void>>().mockResolvedValue();
  update.mockResolvedValue({ ...original, revision: 'revision-two' });
  render(<LangfuseConfigurationForm configuration={original} onSaved={saved} onReload={async () => {}} />);
  const password = screen.getByLabelText('Secret key') as HTMLInputElement;
  expect(password.type).toBe('password');
  expect(password.value).toBe('');
  fireEvent.change(screen.getByLabelText('Host'), { target: { value: 'https://registered.example.test' } });
  expect(password.required).toBe(true);
  fireEvent.change(password, { target: { value: 'replacement-fixture-key' } });
  fireEvent.submit(screen.getByRole('button', { name: 'Save configuration' }).closest('form')!);
  await waitFor(() => expect(saved).toHaveBeenCalled());
  expect(update).toHaveBeenCalledWith({
    host: 'https://registered.example.test', public_key: original.public_key,
    revision: original.revision, secret_key: 'replacement-fixture-key',
  });
  expect(password.value).toBe('');
});

test('leaving the key blank preserves it; removal is explicit and uses the saved destination', async () => {
  update.mockResolvedValue(original);
  render(<LangfuseConfigurationForm configuration={original} onSaved={async () => {}} onReload={async () => {}} />);
  fireEvent.submit(screen.getByRole('button', { name: 'Save configuration' }).closest('form')!);
  await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
  expect(update.mock.calls[0][0]).not.toHaveProperty('secret_key');
  await waitFor(() => expect((screen.getByRole('button', { name: 'Remove saved key' }) as HTMLButtonElement).disabled).toBe(false));
  fireEvent.change(screen.getByLabelText('Host'), { target: { value: 'https://unsaved.example.test' } });
  fireEvent.click(screen.getByRole('button', { name: 'Remove saved key' }));
  await waitFor(() => expect(update).toHaveBeenCalledTimes(2));
  expect(update.mock.calls[1][0]).toEqual({ host: original.host, public_key: original.public_key, revision: original.revision, clear_secret: true });
});

test('a rejected stale edit does not silently retry or report success', async () => {
  const saved = jest.fn<(value: LangfuseConfiguration) => Promise<void>>();
  update.mockRejectedValue(new Error('Langfuse settings changed; reload before saving'));
  render(<LangfuseConfigurationForm configuration={original} onSaved={saved} onReload={async () => {}} />);
  fireEvent.submit(screen.getByRole('button', { name: 'Save configuration' }).closest('form')!);
  await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('reload'));
  expect(update).toHaveBeenCalledTimes(1);
  expect(saved).not.toHaveBeenCalled();
});
