import { FormField } from '@/components/ui/form';
import type { ReactElement } from 'react';
import { FieldArrayPath, type FieldValues, useFieldArray, UseFieldArrayReturn } from 'react-hook-form';

export function FormArrayField<
  TFieldValues extends FieldValues = FieldValues,
  TFieldArrayName extends FieldArrayPath<TFieldValues> = FieldArrayPath<TFieldValues>,
  TKeyName extends string = 'id'
> ({ name, render }: {
  name: TFieldArrayName,
  render: (props: ArrayFieldRenderProps<TFieldValues, TFieldArrayName, TKeyName>) => ReactElement
}) {

  const field = useFieldArray<TFieldValues, TFieldArrayName, TKeyName>({
    name: name,
  });

  return (
    <FormField
      name={name}
      render={() => render({ field })}
    />
  );
}

export type ArrayFieldRenderProps<
  TFieldValues extends FieldValues = FieldValues,
  TFieldArrayName extends FieldArrayPath<TFieldValues> = FieldArrayPath<TFieldValues>,
  TKeyName extends string = 'id'
> = {
  field: UseFieldArrayReturn<TFieldValues, TFieldArrayName, TKeyName>
}
