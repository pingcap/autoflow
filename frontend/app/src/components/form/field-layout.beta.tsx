import type { FormControlWidgetProps } from '@/components/form/control-widget';
import { Button } from '@/components/ui/button';
import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage, useFormContext } from '@/components/ui/form.beta';
import { cn } from '@/lib/utils';
import { type DeepKeys, type DeepValue, type FieldApi, type FormApi, useField } from '@tanstack/react-form';
import { MinusIcon, PlusIcon } from 'lucide-react';
import { type ChangeEvent, cloneElement, type ReactElement, type ReactNode } from 'react';
import { ControllerRenderProps, FieldValues } from 'react-hook-form';

export interface FormFieldLayoutProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends DeepKeys<TFieldValues> = DeepKeys<TFieldValues>
> {
  name: TName;
  label: ReactNode;
  required?: boolean;
  description?: ReactNode;
  // value = props.value ?? fallbackValue
  fallbackValue?: DeepValue<TFieldValues, TName>;
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
      if ((ev as ChangeEvent).type === 'change') {
        field.handleChange((ev as ChangeEvent<HTMLInputElement>).target.value as any);
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
  children,
}: FormFieldLayoutProps<TFieldValues, TName>) {
  return (
    <FormField<TFieldValues, TName>
      name={name}
      render={(field, form, disabled) => (
        <FormItem>
          <FormLabel>
            {label}
            {required && <sup className="text-destructive" aria-hidden>*</sup>}
          </FormLabel>
          <FormControl>
            {renderWidget(children, field, form, disabled, fallbackValue)}
          </FormControl>
          {description && <FormDescription className="break-words">{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
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
  children,
}: FormFieldLayoutProps<TFieldValues, TName>) {
  return (
    <FormField<TFieldValues, TName>
      name={name}
      render={(field, form, disabled) => (
        <FormItem>
          <div className="flex items-center gap-2">
            <FormControl>
              {renderWidget(children, field, form, disabled)}
            </FormControl>
            <FormLabel>{label}</FormLabel>
          </div>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
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
  children,
  unimportant = false,
}: FormFieldLayoutProps<TFieldValues, TName> & { unimportant?: boolean }) {
  return (
    <FormField<TFieldValues, TName>
      name={name}
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
            {renderWidget(children, field, form, disabled, fallbackValue)}
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
}: FormFieldLayoutProps<TFieldValues, TName> & { defaultValue: () => any }) {
  const { form, disabled } = useFormContext<TFieldValues>();
  const arrayField = useField<TFieldValues, TName>({
    name,
    form,
    mode: 'array',
  });

  const arrayFieldValue: any[] = arrayField.state.value as never;

  return (
    <FormField
      name="value"
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
                          {renderWidget(children, field as any, form as any, disabled)}
                        </FormControl>
                        <Button
                          disabled={disabled}
                          size="icon"
                          variant="secondary"
                          type="button"
                          onClick={() => {
                            void arrayField.insertValue(index, defaultValue());
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
              void arrayField.pushValue(defaultValue());
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
