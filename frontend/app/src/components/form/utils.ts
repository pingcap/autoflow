import type { FormApi } from '@tanstack/react-form';
import { z } from 'zod';

export function onSubmitHelper<T> (
  schema: z.ZodType<T>,
  action: (value: T, form: FormApi<T>) => Promise<void>,
  setSubmissionError: (error: unknown) => void,
): (props: { value: T, formApi: FormApi<T> }) => Promise<void> {
  return async ({ value, formApi }) => {
    try {
      setSubmissionError(undefined);
      await action(schema.parse(value), formApi);
    } catch (e) {
      setSubmissionError(e);
    }
  };
}
