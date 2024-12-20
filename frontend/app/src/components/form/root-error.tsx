import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useFormContext as useTanstackFormContext } from '@/components/ui/form.beta';
import { getErrorMessage } from '@/lib/errors';
import type { FormState } from '@tanstack/react-form';
import { useFormContext } from 'react-hook-form';

/**
 * @deprecated
 */
export function FormRootError ({ title = 'Operation failed' }: { title?: string }) {
  const { formState } = useFormContext();
  const error = formState.errors.root;

  return error ? <Alert variant="destructive">
    <AlertTitle>{title}</AlertTitle>
    <AlertDescription>{error.message}</AlertDescription>
  </Alert> : null;
}

export function FormRootErrorBeta ({ title = 'Operation failed' }: { title?: string }) {
  const { form, submissionError } = useTanstackFormContext();

  return (
    <form.Subscribe selector={state => getFormError(state, submissionError)}>
      {(firstError) => !!firstError && (
        <Alert variant="destructive">
          <AlertTitle>{title}</AlertTitle>
          <AlertDescription>{firstError}</AlertDescription>
        </Alert>
      )}
    </form.Subscribe>
  );
}

function getFormError (state: FormState<any>, error: unknown) {
  if (error) {
    return getErrorMessage(error);
  }
  const submitError = state.errorMap.onSubmit;
  if (!submitError) {
    return undefined;
  }
  if (typeof submitError === 'object') {
    return submitError.form;
  }
  return undefined;
}