import { FormRootError } from '@/components/form/root-error';
import { useGeneralSettingsFormContext } from '@/components/settings-form/context';
import { Button } from '@/components/ui/button';
import { Form, formDomEventHandlers } from '@/components/ui/form.beta';
import { getErrorMessage } from '@/lib/errors';
import { useForm } from '@tanstack/react-form';
import { Loader2Icon } from 'lucide-react';
import type { ReactNode } from 'react';
import { z, type ZodType } from 'zod';

export interface GeneralSettingsFieldAccessor<Data, FieldData> {
  path: [keyof Data, ...(string | number | symbol)[]]
  get: (data: Readonly<Data>) => FieldData,
  set: (data: Readonly<Data>, value: FieldData) => Data,
}

export function fieldAccessor<Data, Key extends keyof Data> (key: Key, defaultValue?: Data[Key]): GeneralSettingsFieldAccessor<Data, Data[Key]> {
  return {
    path: [key],
    get: (data) => data[key] ?? defaultValue as Data[Key],
    set: (data, value) => {
      return {
        ...data,
        [key]: value,
      };
    },
  };
}

export function GeneralSettingsField<Data, FieldData> ({
  accessor, schema, children, readonly: fieldReadonly = false,
}: {
  accessor: GeneralSettingsFieldAccessor<Data, FieldData>,
  schema: z.ZodType<FieldData, any, any>,
  readonly?: boolean,
  children: ReactNode,
}) {
  const { data, disabled, readonly, onUpdateField } = useGeneralSettingsFormContext<Data>();
  const form = useForm<{ value: FieldData }>({
    validators: {
      onChange: z.object({
        value: schema,
      }).strict() as ZodType<{ value: FieldData }, any, any>,
      onSubmit: z.object({
        value: schema,
      }).strict() as ZodType<{ value: FieldData }, any, any>,
    },
    defaultValues: {
      value: accessor.get(data),
    },
    onSubmit: async ({ value: { value }, formApi }) => {
      try {
        await onUpdateField(schema.parse(value), accessor);
        formApi.reset({
          value,
        });
      } catch (e) {
        formApi.setErrorMap({
          onChange: getErrorMessage(e),
        });
      }
    },
  });


  return (
    <Form<{ value: FieldData }, undefined> disabled={disabled || readonly || fieldReadonly} form={form}>
      <form className="space-y-6" {...formDomEventHandlers(form)}>
        {children}
        <FormRootError />
        {!readonly && (
          <form.Subscribe selector={state => [state.isDirty, state.isSubmitting] as const}>
            {([isDirty, isSubmitting]) => (isDirty || isSubmitting) && (
              <div className="flex items-center gap-2">
                <Button type="submit" disabled={disabled || isSubmitting || readonly || fieldReadonly}>
                  {isSubmitting && <Loader2Icon className="animate-spin repeat-infinite" />}
                  {isSubmitting ? 'Saving' : 'Save'}
                </Button>
                <Button type="reset" variant="secondary" disabled={disabled || isSubmitting || readonly || fieldReadonly}>Reset</Button>
              </div>
            )}
          </form.Subscribe>
        )}
      </form>
    </Form>
  );
}
