import { FormRootErrorBeta } from '@/components/form/root-error';
import { Form as FormBeta, formDomEventHandlers, FormSubmit } from '@/components/ui/form.beta';
import { useForm as useTanstackForm } from '@tanstack/react-form';
import { type ReactNode, useId, useState } from 'react';
import { z } from 'zod';

export interface CreateEntityFormBetaProps<T, R> {
  defaultValues?: T;
  onCreated?: (data: R) => void;
  onInvalid?: () => void;
  transitioning?: boolean;
  children?: ReactNode;
}

export function withCreateEntityFormBeta<T, R> (
  schema: z.ZodType<T, any, any>,
  createApi: (data: T) => Promise<R>,
  { submitTitle = 'Create', submittingTitle }: {
    submitTitle?: ReactNode
    submittingTitle?: ReactNode
  } = {},
) {
  return function CreateEntityFormBeta (
    {
      defaultValues,
      onCreated,
      onInvalid,
      transitioning,
      children,
    }: CreateEntityFormBetaProps<T, R>,
  ) {
    const id = useId();
    const [submissionError, setSubmissionError] = useState<unknown>();

    const form = useTanstackForm<T>({
      validators: {
        onSubmit: schema,
      },
      defaultValues,
      onSubmit: async ({ value, formApi }) => {
        try {
          const data = await createApi(schema.parse(value));
          onCreated?.(data);
        } catch (e) {
          setSubmissionError(e);
        }
      },
      onSubmitInvalid: () => {
        onInvalid?.();
      },
    });

    return (
      <FormBeta form={form} disabled={transitioning} submissionError={submissionError}>
        <form
          id={id}
          className="max-w-screen-sm space-y-4"
          {...formDomEventHandlers(form, transitioning)}
        >
          {children}
          <FormRootErrorBeta />
          <FormSubmit form={id} transitioning={transitioning} submittingChildren={submittingTitle}>
            {submitTitle}
          </FormSubmit>
        </form>
      </FormBeta>
    );
  };
}

export { withCreateEntityFormBeta as withCreateEntityForm };
