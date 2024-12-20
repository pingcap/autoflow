import type { FormControlWidgetProps } from '@/components/form/control-widget';
import { Button } from '@/components/ui/button';
import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage, useFormContext } from '@/components/ui/form.beta';
import { isChangeEvent } from '@/lib/react';
import { cn } from '@/lib/utils';
import { type DeepKeys, type DeepValue, type FieldApi, FieldValidators, type FormApi, useField } from '@tanstack/react-form';
import { MinusIcon, PlusIcon } from 'lucide-react';
import { cloneElement, type ReactElement, type ReactNode } from 'react';

type WidgetProps<TFormData, TName extends DeepKeys<TFormData>> = Required<Omit<FormControlWidgetProps<DeepValue<TFormData, TName>>, 'id' | 'aria-invalid' | 'aria-describedby'>>

type DefaultFormData = Record<string, any>

export interface FormFieldLayoutProps<
  TFormData extends DefaultFormData,
  TName extends DeepKeys<TFormData> = DeepKeys<TFormData>
> {
  name: TName;
  label: ReactNode;
  required?: boolean;
  description?: ReactNode;
  /**
   * Fallback value is used for display. This value will not submit to server.
   */
  fallbackValue?: DeepValue<TFormData, TName>;
  defaultValue?: NoInfer<DeepValue<TFormData, TName>>;
  validators?: FieldValidators<TFormData, TName>;

  children: ((props: WidgetProps<TFormData, TName>) => ReactNode) | ReactElement<WidgetProps<TFormData, TName>>;
}

function renderWidget<
  TFormData extends DefaultFormData,
  TName extends DeepKeys<TFormData> = DeepKeys<TFormData>
> (
  children: FormFieldLayoutProps<TFormData, TName>['children'],
  field: FieldApi<TFormData, TName>,
  form: FormApi<TFormData>,
  disabled: boolean | undefined,
  fallbackValue?: DeepValue<TFormData, TName>,
) {

  const data: WidgetProps<TFormData, TName> = {
    value: field.state.value ?? fallbackValue as any,
    name: field.name as TName,
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
    ref: () => {},
  };

  if (typeof children === 'function') {
    return children(data);
  } else {
    return cloneElement(children, data);
  }
}

export function FormFieldBasicLayout<
  TFormData extends DefaultFormData,
  TName extends DeepKeys<TFormData> = DeepKeys<TFormData>
> ({
  name,
  label,
  description,
  required,
  fallbackValue,
  defaultValue,
  validators,
  children,
}: FormFieldLayoutProps<TFormData, TName>) {
  return (
    <FormField<TFormData, TName>
      name={name}
      defaultValue={defaultValue}
      render={(field, form, disabled) => (
        <FormItem>
          <FormLabel>
            {label}
            {required && <sup className="text-destructive" aria-hidden>*</sup>}
          </FormLabel>
          <FormControl>
            {renderWidget<TFormData, TName>(children, field, form, disabled, fallbackValue)}
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
  TFormData extends DefaultFormData,
  TName extends DeepKeys<TFormData> = DeepKeys<TFormData>
> ({
  name,
  label,
  description,
  defaultValue,
  validators,
  children,
}: FormFieldLayoutProps<TFormData, TName>) {
  return (
    <FormField<TFormData, TName>
      name={name}
      defaultValue={defaultValue}
      render={(field, form, disabled) => (
        <FormItem>
          <div className="flex items-center gap-2">
            <FormControl>
              {renderWidget<TFormData, TName>(children, field, form, disabled)}
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
  TFormData extends DefaultFormData,
  TName extends DeepKeys<TFormData> = DeepKeys<TFormData>
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
}: FormFieldLayoutProps<TFormData, TName> & { unimportant?: boolean }) {
  return (
    <FormField<TFormData, TName>
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
            {renderWidget<TFormData, TName>(children, field, form, disabled, fallbackValue)}
          </FormControl>
        </FormItem>
      )}
    />
  );
}

type DeepKeysOfType<T, Value> = string & keyof { [P in DeepKeys<T> as DeepValue<T, P> extends Value ? P : never]: any }

export function FormPrimitiveArrayFieldBasicLayout<
  TFormData extends DefaultFormData,
  TName extends DeepKeysOfType<TFormData, any[]> = DeepKeysOfType<TFormData, any[]>
> ({
  name,
  label,
  description,
  children,
  required,
  defaultValue,
  validators,
  newItemValue,
}: FormFieldLayoutProps<TFormData, TName> & { newItemValue: () => any }) {
  const { form } = useFormContext<TFormData>();
  const arrayField = useField<TFormData, TName>({
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
