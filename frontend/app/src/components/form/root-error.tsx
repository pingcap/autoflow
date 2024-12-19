import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useFormContext as useTanstackFormContext } from '@/components/ui/form.beta';
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
  const { form } = useTanstackFormContext();

  return (
    <form.Subscribe
      selector={state => [state.errors[0]] as const}
    >
      {([firstError]) => !!firstError && (
        <Alert variant="destructive">
          <AlertTitle>{title}</AlertTitle>
          <AlertDescription>{firstError}</AlertDescription>
        </Alert>
      )}
    </form.Subscribe>
  );
}
