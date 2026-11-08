import { jest } from '@jest/globals';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { z } from 'zod';

const updateSetting = jest.fn<(name: string, value: unknown) => Promise<void>>().mockResolvedValue();
jest.unstable_mockModule('../../api/site-settings', () => ({ updateSiteSetting: updateSetting }));
jest.unstable_mockModule('next/navigation', () => ({ useRouter: () => ({ refresh () {} }) }));
const { SettingsField } = await import('../settings/SettingsField');
const { withCreateEntityForm } = await import('./create-entity-form');
const { FormInput } = await import('./control-widget');

afterEach(() => { cleanup(); jest.clearAllMocks(); });

test('React Hook Form keeps integer validation and submits the parsed numeric value', async () => {
  render(<SettingsField name="count" item={{ name: 'count', data_type: 'int', default: 1, value: 1, description: 'Count', group: 'test', client: false }} />);
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '1.5' } });
  fireEvent.click(await screen.findByRole('button', { name: 'Save' }));
  await screen.findByText(/integer|int/i);
  expect(updateSetting).not.toHaveBeenCalled();
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '3' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));
  await waitFor(() => expect(updateSetting).toHaveBeenCalledWith('count', 3));
});

test('TanStack forms reject invalid values and keep standard-schema numeric input/output', async () => {
  const create = jest.fn<(value: { count: number }) => Promise<{ count: number }>>().mockImplementation(async value => value);
  const invalid = jest.fn();
  const TestForm = withCreateEntityForm(z.object({ count: z.coerce.number<number>().int().min(1) }), create);
  render(<TestForm defaultValues={{ count: 0 }} onInvalid={invalid}>
    <TestForm.Basic name="count" label="Count"><FormInput type="number" /></TestForm.Basic>
  </TestForm>);
  fireEvent.submit(screen.getByRole('button', { name: 'Create' }).closest('form')!);
  await waitFor(() => expect(invalid).toHaveBeenCalled());
  expect(create).not.toHaveBeenCalled();
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '3' } });
  fireEvent.submit(screen.getByRole('button', { name: 'Create' }).closest('form')!);
  await waitFor(() => expect(create).toHaveBeenCalledWith({ count: 3 }));
});
