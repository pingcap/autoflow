// This file contains new form components based on @tanstack/form
// The components should be aligned with original form components.

import { useRegisterFieldInFormSection } from '@/components/form-sections';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import { Slot } from '@radix-ui/react-slot';
import type { FieldValidators } from '@tanstack/react-form';
import * as FormPrimitives from '@tanstack/react-form';
import { type DeepValue, type FieldApi, type FormApi, type NoInfer, type ReactFormExtendedApi, useField } from '@tanstack/react-form';
import { Loader2Icon } from 'lucide-react';
import * as React from 'react';
import { type ComponentProps, createContext, type FormEvent, type ReactNode, useContext, useId } from 'react';

const FormContext = createContext<{
  form: FormPrimitives.ReactFormExtendedApi<any, any>
  disabled?: boolean
} | undefined
>(undefined);

function useFormContext<
  TFormData,
  TFormValidator extends FormPrimitives.Validator<TFormData, unknown> | undefined = undefined,
> () {
  const api = useContext(FormContext);
  if (!api) {
    throw new Error('Require tanstack form context');
  }
  return {
    ...api,
    form: api.form as FormPrimitives.ReactFormExtendedApi<TFormData, TFormValidator>,
  };
}

function formDomEventHandlers (form: FormApi<any>, disabled?: boolean): Pick<ComponentProps<'form'>, 'onSubmit' | 'onReset'> {
  return {
    onSubmit: (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (!disabled) {
        void form.handleSubmit();
      }
    },
    onReset: (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (!disabled) {
        void form.reset();
      }
    },
  };
}

const FormFieldContext = createContext<{ name: any, mode?: 'value' | 'array' | undefined } | undefined>(undefined);
const FormItemContext = createContext<{ id: string } | undefined>(undefined);

function useFormField<
  TFormData,
  TName extends FormPrimitives.DeepKeys<TFormData>,
  TFormValidator extends FormPrimitives.Validator<TFormData, unknown> | undefined = undefined,
> () {
  const { form } = useFormContext<TFormData, TFormValidator>();
  const fieldContext = useContext(FormFieldContext);
  const itemContext = useContext(FormItemContext);

  if (!itemContext) {
    throw new Error('useFormField() should be used within <FormItem>');
  }

  if (!fieldContext) {
    throw new Error('useFormField() should be used within <FormField>');
  }

  const field = form.getFieldMeta(fieldContext.name as TName);

  const { id } = itemContext;

  return {
    id: id,
    name: fieldContext.name as TName,
    formItemId: `${id}-form-item`,
    formDescriptionId: `${id}-form-item-description`,
    formMessageId: `${id}-form-item-message`,
    field,
  };
}

interface FormProps<
  TFormData,
  TFormValidator extends FormPrimitives.Validator<TFormData, unknown> | undefined = undefined,
> {
  form: FormPrimitives.ReactFormExtendedApi<TFormData, TFormValidator>;
  disabled?: boolean;
  children: ReactNode;
}

function Form<
  TFormData,
  TFormValidator extends FormPrimitives.Validator<TFormData, unknown> | undefined = undefined,
> ({ children, disabled, form }: FormProps<TFormData, TFormValidator>) {
  return (
    <FormContext value={{ form: form as FormPrimitives.ReactFormExtendedApi<TFormData, TFormValidator>, disabled }}>
      {children}
    </FormContext>
  );
}

function FormField<
  TFormData,
  TName extends FormPrimitives.DeepKeys<TFormData>,
  TFieldValidator extends FormPrimitives.Validator<DeepValue<TFormData, TName>, unknown> | undefined = undefined,
  TFormValidator extends FormPrimitives.Validator<TFormData, unknown> | undefined = undefined,
> ({ name, defaultValue, validators, mode, render }: {
  name: TName
  defaultValue?: NoInfer<DeepValue<TFormData, TName>>
  mode?: 'value' | 'array' | undefined
  validators?: FieldValidators<TFormData, TName, TFieldValidator, TFormValidator>;
  render: (
    field: FieldApi<TFormData, TName, TFieldValidator, TFormValidator, FormPrimitives.DeepValue<TFormData, TName>>,
    form: ReactFormExtendedApi<TFormData, TFormValidator>,
    disabled: boolean | undefined,
  ) => ReactNode
}) {
  const { form, disabled } = useFormContext<TFormData, TFormValidator>();

  const field = useField<TFormData, TName, TFieldValidator, TFormValidator>({
    form,
    name,
    mode,
    defaultValue,
    validators,
  });

  useRegisterFieldInFormSection(field);

  return (
    <FormFieldContext value={{ name, mode }}>
      {render(field, form, disabled)}
    </FormFieldContext>
  );
}

function FormItem ({ className, ref, ...props }: ComponentProps<'div'>) {
  const _id = useId();
  const id = props.id ?? _id;
  return (
    <FormItemContext.Provider value={{ id }}>
      <div ref={ref} className={cn('space-y-2', className)} {...props} />
    </FormItemContext.Provider>
  );
}

function FormLabel ({ ref, className, ...props }: ComponentProps<typeof Label>) {
  const { field, formItemId } = useFormField();
  const error = !!field?.errors?.length;

  return (
    <Label
      ref={ref}
      className={cn(error && 'text-destructive', className)}
      htmlFor={formItemId}
      {...props}
    />
  );
}

function FormControl ({ ref, ...props }: ComponentProps<typeof Slot>) {
  const { field, formItemId, formDescriptionId, formMessageId } = useFormField();
  const error = field?.errors?.[0];

  return (
    <Slot
      ref={ref}
      id={formItemId}
      aria-describedby={
        !error
          ? `${formDescriptionId}`
          : `${formDescriptionId} ${formMessageId}`
      }
      aria-invalid={!!error}
      {...props}
    />
  );
}

function FormDescription ({ ref, className, ...props }: ComponentProps<'p'>) {
  const { formDescriptionId } = useFormField();

  return (
    <p
      ref={ref}
      id={formDescriptionId}
      className={cn('text-sm text-muted-foreground', className)}
      {...props}
    />
  );
}

function FormMessage ({ ref, className, children, ...props }: ComponentProps<'p'>) {
  const { field, formMessageId } = useFormField();
  const error = field?.errors?.[0];
  const body = error ? String(error) : children;

  if (!body) {
    return null;
  }

  return (
    <p
      ref={ref}
      id={formMessageId}
      className={cn('text-sm font-medium text-destructive', className)}
      {...props}
    >
      {body}
    </p>
  );
}

function FormSubmit ({
  children,
  asChild,
  disabled,
  transitioning,
  ...props
}: Omit<ComponentProps<typeof Button>, 'formAction' | 'type'> & {
  /*
   * Used when to start a transition after created an entity. The loader indicator will be shown while transitioning.
   */
  transitioning?: boolean
}) {
  const { form } = useFormContext();

  return (
    <Button {...props} type="submit" disabled={form.state.isSubmitting || transitioning || disabled}>
      {asChild
        ? children
        : (form.state.isSubmitting || transitioning)
          ? <>
            <Loader2Icon className="animate-spin repeat-infinite" />
            {children}
          </>
          : children}
    </Button>
  );
}

export { useFormContext, Form, FormField, FormItem, FormLabel, FormControl, FormMessage, FormDescription, FormSubmit, formDomEventHandlers };
