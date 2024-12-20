import type { FormControlWidgetProps } from '@/components/form/control-widget';
import { Button } from '@/components/ui/button';
import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage, useFormContext } from '@/components/ui/form.beta';
import { isChangeEvent } from '@/lib/react';
import { cn } from '@/lib/utils';
import { type DeepKeys, type DeepValue, type FieldApi, FieldValidators, type FormApi, useField } from '@tanstack/react-form';
import { MinusIcon, PlusIcon } from 'lucide-react';
import { cloneElement, type ReactElement, type ReactNode } from 'react';
import { ControllerRenderProps, FieldValues } from 'react-hook-form';

export interface FormFieldLayoutProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends DeepKeys<TFieldValues> = DeepKeys<TFieldValues>
> {
  name: TName;
  label: ReactNode;
  required?: boolean;
  description?: ReactNode;
  /**
   * Fallback value is used for display. This value will not submit to server.
   */
  fallbackValue?: DeepValue<TFieldValues, TName>;
  defaultValue?: NoInfer<DeepValue<TFieldValues, TName>>;
  validators?: FieldValidators<TFieldValues, TName>;

  children: ((props: ControllerRenderProps<TFieldValues, any>) => ReactNode) | ReactElement<FormControlWidgetProps<TFieldValues, any>>;
}

function renderWidget<
  TFieldValues extends FieldValues = FieldValues,
  TName extends DeepKeys<TFieldValues> = DeepKeys<TFieldValues>
> (
  children: FormFieldLayoutProps<TFieldValues, TName>['children'],
  field: FieldApi<TFieldValues, TName>,
  form: FormApi<TFieldValues>,
  disabled: boolean | undefined,
  fallbackValue?: DeepValue<TFieldValues, TName>,
) {

  const data = {
    value: field.state.value ?? fallbackValue as any,
    name: field.name,
    onChange: ((ev: any) => {
      if (isChangeEvent(ev)) {
        const el = ev.currentTarget;
        if (el instanceof HTMLInputElement) {
          if (el.type === 'number') {
            field.handleChange(el.valueAsNumber as any);
            return;
          } else if (el.type === 'date' || el.type === 'datetime-local') {
            field.handleChange(el.valueAsDate as any);
            return;
          }
        }
        field.handleChange((el as HTMLInputElement).value as any);
      } else {
        field.handleChange(ev);
      }
    }),
    onBlur: field.handleBlur,
    disabled: disabled || field.form.state.isSubmitting,
    ref: field.mount,
  };

  if (typeof children === 'function') {
    return children(data);
  } else {
    return cloneElement(children, data);
  }
}

export function FormFieldBasicLayout<
  TFieldValues extends FieldValues = FieldValues,
  TName extends DeepKeys<TFieldValues> = DeepKeys<TFieldValues>
> ({
  name,
  label,
  description,
  required,
  fallbackValue,
  defaultValue,
  validators,
  children,
}: FormFieldLayoutProps<TFieldValues, TName>) {
  return (
    <FormField<TFieldValues, TName>
      name={name}
      defaultValue={defaultValue}
      render={(field, form, disabled) => (
        <FormItem>
          <FormLabel>
            {label}
            {required && <sup className="text-destructive" aria-hidden>*</sup>}
          </FormLabel>
          <FormControl>
            {renderWidget<TFieldValues, TName>(children, field, form, disabled, fallbackValue)}
          </FormControl>
          {description && <FormDescription className="break-words">{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
      validators={validators}
    />
  );
}

export function FormFieldInlineLayout<
  TFieldValues extends FieldValues = FieldValues,
  TName extends DeepKeys<TFieldValues> = DeepKeys<TFieldValues>
> ({
  name,
  label,
  description,
  defaultValue,
  validators,
  children,
}: FormFieldLayoutProps<TFieldValues, TName>) {
  return (
    <FormField<TFieldValues, TName>
      name={name}
      defaultValue={defaultValue}
      render={(field, form, disabled) => (
        <FormItem>
          <div className="flex items-center gap-2">
            <FormControl>
              {renderWidget<TFieldValues, TName>(children, field, form, disabled)}
            </FormControl>
            <FormLabel>{label}</FormLabel>
          </div>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
      validators={validators}
    />
  );
}

export function FormFieldContainedLayout<
  TFieldValues extends FieldValues = FieldValues,
  TName extends DeepKeys<TFieldValues> = DeepKeys<TFieldValues>
> ({
  name,
  label,
  description,
  required,
  fallbackValue,
  defaultValue,
  validators,
  children,
  unimportant = false,
}: FormFieldLayoutProps<TFieldValues, TName> & { unimportant?: boolean }) {
  return (
    <FormField<TFieldValues, TName>
      name={name}
      defaultValue={defaultValue}
      validators={validators}
      render={(field, form, disabled) => (
        <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
          <div className="space-y-0.5">
            <FormLabel className={cn(!unimportant && 'text-base')}>
              {label}
              {required && <sup className="text-destructive" aria-hidden>*</sup>}
            </FormLabel>
            {description && <FormDescription>
              {description}
            </FormDescription>}
          </div>
          <FormControl>
            {renderWidget<TFieldValues, TName>(children, field, form, disabled, fallbackValue)}
          </FormControl>
        </FormItem>
      )}
    />
  );
}

type DeepKeysOfType<T, Value> = string & keyof { [P in DeepKeys<T> as DeepValue<T, P> extends Value ? P : never]: any }

export function FormPrimitiveArrayFieldBasicLayout<
  TFieldValues extends FieldValues = FieldValues,
  TName extends DeepKeysOfType<TFieldValues, any[]> = DeepKeysOfType<TFieldValues, any[]>
> ({
  name,
  label,
  description,
  children,
  required,
  defaultValue,
  validators,
  newItemValue,
}: FormFieldLayoutProps<TFieldValues, TName> & { newItemValue: () => any }) {
  const { form } = useFormContext<TFieldValues>();
  const arrayField = useField<TFieldValues, TName>({
    name,
    form,
    mode: 'array',
  });

  const arrayFieldValue: any[] = arrayField.state.value as never;

  return (
    <FormField
      name={name}
      defaultValue={defaultValue}
      validators={validators}
      render={() => (
        <FormItem>
          <FormLabel>
            {label}
            {required && <sup className="text-destructive" aria-hidden>*</sup>}
          </FormLabel>
          <ol className="space-y-2">
            {arrayFieldValue.map((_, index) => (
              <FormField
                key={index}
                name={`${name}[${index}]`}
                render={(field, form, disabled) => (
                  <li>
                    <FormItem>
                      <div className="flex gap-2">
                        <FormControl className="flex-1">
                          {renderWidget<any, any>(children, field as any, form as any, disabled)}
                        </FormControl>
                        <Button
                          disabled={disabled}
                          size="icon"
                          variant="secondary"
                          type="button"
                          onClick={() => {
                            void arrayField.insertValue(index, newItemValue());
                          }}
                        >
                          <PlusIcon className="size-4" />
                        </Button>
                        <Button
                          disabled={disabled}
                          size="icon"
                          variant="ghost"
                          type="button"
                          onClick={() => {
                            void arrayField.removeValue(index);
                          }}
                        >
                          <MinusIcon className="size-4" />
                        </Button>
                      </div>
                      <FormMessage />
                    </FormItem>
                  </li>
                )}
              />
            ))}
          </ol>
          <Button
            className="w-full"
            variant="outline"
            type="button"
            onClick={() => {
              void arrayField.pushValue(newItemValue());
            }}
          >
            <PlusIcon className="w-4 mr-1" />
            New Item
          </Button>
          {description && <FormDescription className="break-words">{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
