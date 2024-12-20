import { formFieldLayout, type TypedFormFieldLayouts } from '@/components/form/field-layout';
import { FormRootErrorBeta } from '@/components/form/root-error';
import { Form as FormBeta, formDomEventHandlers, FormSubmit } from '@/components/ui/form.beta';
import { useForm as useTanstackForm } from '@tanstack/react-form';
import { type FunctionComponent, type ReactNode, useId, useState } from 'react';
import { z } from 'zod';

export interface CreateEntityFormBetaProps<R, I> {
  defaultValues?: I;
  onCreated?: (data: R) => void;
  onInvalid?: () => void;
  transitioning?: boolean;
  children?: ReactNode;
}

interface CreateEntityFormComponent<R, I> extends FunctionComponent<CreateEntityFormBetaProps<R, I>>, TypedFormFieldLayouts<I> {
}

export function withCreateEntityFormBeta<T, R, I = any> (
  schema: z.ZodType<T, any, I>,
  createApi: (data: T) => Promise<R>,
  { submitTitle = 'Create', submittingTitle }: {
    submitTitle?: ReactNode
    submittingTitle?: ReactNode
  } = {},
): CreateEntityFormComponent<R, I> {

  function CreateEntityFormBeta (
    {
      defaultValues,
      onCreated,
      onInvalid,
      transitioning,
      children,
    }: CreateEntityFormBetaProps<R, I>,
  ) {
    const id = useId();
    const [submissionError, setSubmissionError] = useState<unknown>();

    const form = useTanstackForm<I>({
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
  }

  Object.assign(CreateEntityFormBeta, formFieldLayout<I>());

  return CreateEntityFormBeta as CreateEntityFormComponent<R, I>;
}

export { withCreateEntityFormBeta as withCreateEntityForm };
