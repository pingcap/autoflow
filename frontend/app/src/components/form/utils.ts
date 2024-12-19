import { getErrorMessage } from '@/lib/errors';
import type { FormApi } from '@tanstack/react-form';
import { FormEvent } from 'react';
import type { UseFormReturn } from 'react-hook-form';
import { z } from 'zod';

export function handleSubmitHelper<Form extends UseFormReturn<any, any, any>> (
  form: Form,
  onValid: Parameters<Form['handleSubmit']>[0],
  onInvalid?: Parameters<Form['handleSubmit']>[1],
) {
  const handleSubmit = form.handleSubmit(
    async (data) => {
      try {
        await onValid(data);
      } catch (e) {
        form.setError('root', { message: getErrorMessage(e) });
      }
    },
    onInvalid,
  );

  return (event: FormEvent<HTMLFormElement>) => {
    // Prevent submit event being propagated to parent react components.
    event.stopPropagation();
    void handleSubmit(event);
  };
}

export function onSubmitHelper<T> (schema: z.ZodType<T>, action: (value: T, form: FormApi<T>) => Promise<void>): (props: { value: T, formApi: FormApi<T> }) => Promise<void> {
  return async ({ value, formApi }) => {
    try {
      await action(schema.parse(value), formApi);
    } catch (e) {
      formApi.setErrorMap({
        onSubmit: getErrorMessage(e),
      });
    }
  };
}
