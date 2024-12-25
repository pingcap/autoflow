import { FormRootError } from '@/components/form/root-error';
import { handleSubmitHelper } from '@/components/form/utils';
import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { zodResolver } from '@hookform/resolvers/zod';
import { type ReactNode, useId } from 'react';
import { type DefaultValues, type FieldValues, useForm } from 'react-hook-form';
import { z } from 'zod';

export interface CreateEntityFormProps<T extends FieldValues, R> {
  defaultValues?: DefaultValues<T>;
  onCreated?: (data: R) => void;
  transitioning?: boolean;
  children?: ReactNode;
}

export function withCreateEntityForm<T extends FieldValues, R> (
  schema: z.ZodType<T, any, any>,
  createApi: (data: T) => Promise<R>,
) {
  const resolver = zodResolver(schema);

  return function CreateEntityForm (
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
  };
}
