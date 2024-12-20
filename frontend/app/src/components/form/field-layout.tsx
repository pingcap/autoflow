export * from './field-layout.beta';

import type { CreateEntityFormBetaProps } from '@/components/form/create-entity-form';
import type { DeepKeys } from '@tanstack/react-form';
import type { ComponentProps, ComponentType, ReactNode } from 'react';
import { z } from 'zod';
import { type DeepKeysOfType, FormFieldBasicLayout, FormFieldContainedLayout, FormFieldInlineLayout, FormPrimitiveArrayFieldBasicLayout } from './field-layout.beta';

/**
 * This function creates typed form layout components.
 *
 * - If T is ZodType, TFormData is the input
 * - If T is {@link CreateEntityFormBetaProps} or return type of {@link import('@/components/form/create-entity-form').withCreateEntityFormBeta}, TFormData is the form input type
 * - If T is Record<string, any>, TFormData is itself
 */
export function formFieldLayout<T> () {
  type TFormData =
    T extends z.ZodType<any, any, any>
      ? z.input<T>
      : T extends CreateEntityFormBetaProps<any, infer I>
        ? I
        : T extends ComponentType<CreateEntityFormBetaProps<any, infer I>>
          ? I
          : T extends Record<string, any>
            ? T
            : never;

  return {
    Basic: FormFieldBasicLayout as <TName extends DeepKeys<TFormData>> (props: ComponentProps<typeof FormFieldBasicLayout<TFormData, TName>>) => ReactNode,
    Contained: FormFieldContainedLayout as <TName extends DeepKeys<TFormData>> (props: ComponentProps<typeof FormFieldContainedLayout<TFormData, TName>>) => ReactNode,
    Inline: FormFieldInlineLayout as <TName extends DeepKeys<TFormData>> (props: ComponentProps<typeof FormFieldInlineLayout<TFormData, TName>>) => ReactNode,
    PrimitiveArray: FormPrimitiveArrayFieldBasicLayout as <TName extends DeepKeysOfType<TFormData, any[]>> (props: ComponentProps<typeof FormPrimitiveArrayFieldBasicLayout<TFormData, TName>>) => ReactNode,
  } satisfies TypedFormFieldLayouts<TFormData>;
}

export interface TypedFormFieldLayouts<TFormData> {
  Basic: <TName extends DeepKeys<TFormData>> (props: ComponentProps<typeof FormFieldBasicLayout<TFormData, TName>>) => ReactNode,
  Contained: <TName extends DeepKeys<TFormData>> (props: ComponentProps<typeof FormFieldContainedLayout<TFormData, TName>>) => ReactNode,
  Inline: <TName extends DeepKeys<TFormData>> (props: ComponentProps<typeof FormFieldInlineLayout<TFormData, TName>>) => ReactNode,
  PrimitiveArray: <TName extends DeepKeysOfType<TFormData, any[]>> (props: ComponentProps<typeof FormPrimitiveArrayFieldBasicLayout<TFormData, TName>>) => ReactNode,
}
