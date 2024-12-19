import type { FormControlWidgetProps } from '@/components/form/control-widget';
import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { cn } from '@/lib/utils';
import { cloneElement, type ReactElement, type ReactNode } from 'react';
import { ControllerRenderProps, FieldPath, type FieldPathValue, FieldValues } from 'react-hook-form';

export interface FormFieldLayoutProps<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>
> {
  name: TName;
  label: ReactNode;
  required?: boolean;
  description?: ReactNode;
  // value = props.value ?? fallbackValue
  fallbackValue?: FieldPathValue<TFieldValues, TName>;
  children: ((props: ControllerRenderProps<TFieldValues, TName>) => ReactNode) | ReactElement<FormControlWidgetProps<TFieldValues, TName>>;
}

function renderWidget<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>
> (children: FormFieldLayoutProps<TFieldValues, TName>['children'], { value, ...props }: ControllerRenderProps<TFieldValues, TName>, fallbackValue?: FieldPathValue<TFieldValues, TName>) {
  if (typeof children === 'function') {
    return children({ value: value ?? fallbackValue as never, ...props });
  } else {
    return cloneElement(children, { value: value ?? fallbackValue as never, ...props });
  }
}

/**
 * @deprecated
 */
export function FormFieldBasicLayout<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>
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
      render={({ field }) => (
        <FormItem>
          <FormLabel>
            {label}
            {required && <sup className="text-destructive" aria-hidden>*</sup>}
          </FormLabel>
          <FormControl>
            {renderWidget(children, field, fallbackValue)}
          </FormControl>
          {description && <FormDescription className="break-words">{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

/**
 * @deprecated
 */
export function FormFieldInlineLayout<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>
> ({
  name,
  label,
  description,
  children,
}: FormFieldLayoutProps<TFieldValues, TName>) {
  return (
    <FormField<TFieldValues, TName>
      name={name}
      render={({ field }) => (
        <FormItem>
          <div className="flex items-center gap-2">
            <FormControl>
              {renderWidget(children, field)}
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

/**
 * @deprecated
 */
export function FormFieldContainedLayout<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>
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
      render={({ field }) => (
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
            {renderWidget(children, field, fallbackValue)}
          </FormControl>
        </FormItem>
      )}
    />
  );
}
