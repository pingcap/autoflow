import { formFieldLayout, type TypedFormFieldLayouts } from '@/components/form/field-layout';
import { FormRootError } from '@/components/form/root-error';
import { Form, formDomEventHandlers, FormSubmit } from '@/components/ui/form.beta';
import { useForm } from '@tanstack/react-form';
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

export function withCreateEntityForm<T, R, I = any> (
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

    const form = useForm<I>({
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
      <Form form={form} disabled={transitioning} submissionError={submissionError}>
        <form
          id={id}
          className="max-w-screen-sm space-y-4"
          {...formDomEventHandlers(form, transitioning)}
        >
          {children}
          <FormRootError />
          <FormSubmit form={id} transitioning={transitioning} submittingChildren={submittingTitle}>
            {submitTitle}
          </FormSubmit>
        </form>
      </Form>
    );
  }

  Object.assign(CreateEntityFormBeta, formFieldLayout<I>());

  return CreateEntityFormBeta as CreateEntityFormComponent<R, I>;
}
