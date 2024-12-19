import { FormRootError, FormRootErrorBeta } from '@/components/form/root-error';
import { handleSubmitHelper } from '@/components/form/utils';
import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { Form as FormBeta, formDomEventHandlers } from '@/components/ui/form.beta';
import { getErrorMessage } from '@/lib/errors';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm as useTanstackForm } from '@tanstack/react-form';
import { type ReactNode, useId } from 'react';
import { type DefaultValues, type FieldValues, useForm } from 'react-hook-form';
import { z } from 'zod';

/**
 * @deprecated
 */
export interface CreateEntityFormProps<T extends FieldValues, R> {
  defaultValues?: DefaultValues<T>;
  onCreated?: (data: R) => void;
  transitioning?: boolean;
  children?: ReactNode;
}

/**
 * @deprecated
 */
export function withCreateEntityForm<T extends FieldValues, R> (
  schema: z.ZodType<T, any, any>,
  createApi: (data: T) => Promise<R>,
) {
  const resolver = zodResolver(schema);

  /**
   * @deprecated
   */
  function CreateEntityForm (
    {
      defaultValues,
      onCreated,
      transitioning,
      children,
    }: CreateEntityFormProps<T, R>,
  ) {
    const id = useId();

    const form = useForm<T>({
      resolver,
      defaultValues,
    });

    const handleSubmit = handleSubmitHelper(form, async data => {
      const result = await createApi(data);

      onCreated?.(result);
    });

    return (
      <Form {...form}>
        <form id={id} className="max-w-screen-sm space-y-4" onSubmit={handleSubmit}>
          {children}
          <FormRootError />
          <Button type="submit" form={id} disabled={transitioning || form.formState.isSubmitting || form.formState.disabled}>
            Create
          </Button>
        </form>
      </Form>
    );
  }

  return CreateEntityForm;
}

export interface CreateEntityFormBetaProps<T, R> {
  defaultValues?: T;
  onCreated?: (data: R) => void;
  transitioning?: boolean;
  children?: ReactNode;
}

export function withCreateEntityFormBeta<T, R> (
  schema: z.ZodType<T, any, any>,
  createApi: (data: T) => Promise<R>,
) {
  return function CreateEntityFormBeta (
    {
      defaultValues,
      onCreated,
      transitioning,
      children,
    }: CreateEntityFormBetaProps<T, R>,
  ) {
    const id = useId();

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
          formApi.setErrorMap({
            onSubmit: getErrorMessage(e),
          });
        }
      },
    });

    return (
      <FormBeta form={form} disabled={transitioning}>
        <form
          id={id}
          className="max-w-screen-sm space-y-4"
          {...formDomEventHandlers(form, transitioning)}
        >
          {children}
          <FormRootErrorBeta />
          <Button type="submit" form={id} disabled={transitioning || form.state.isSubmitting}>
            Create
          </Button>
        </form>
      </FormBeta>
    );
  };
}
