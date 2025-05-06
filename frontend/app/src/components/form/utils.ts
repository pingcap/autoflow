import type { FieldInfo, FormApi, ValidationErrorMap } from '@tanstack/react-form';
import { z, ZodError } from 'zod';

export function onSubmitHelper<T> (
  schema: z.ZodType<T>,
  action: (value: T, form: FormApi<T>) => Promise<void>,
  setSubmissionError: (error: unknown) => void,
): (props: { value: T, formApi: FormApi<T> }) => Promise<void> {
  return async ({ value, formApi }) => {
    try {
      setSubmissionError(undefined);
      await action(schema.parse(value), formApi);
    } catch (e) {
      if (e != null && e instanceof ZodError) {
        const { formErrors, fieldErrors } = e.flatten();
        const rest = applyFormError(formApi, Object.assign(
          {} as Record<string, string[]>,
          fieldErrors,
          formErrors.length > 0 ? {
            '.': formErrors,
          } : {},
        ), 'onSubmit');
        if (rest) {
          setSubmissionError(Object.values(rest).join(' '));
        }
      } else {
        setSubmissionError(e);
      }
    }
  };
}

/**
 * Applies error messages to the appropriate fields in the given form API based on the provided error body.
 * Matches each error message to its corresponding field within the form using the specified validation phase.
 * Returns any unhandled errors that do not correspond to fields in the form.
 *
 * @see https://github.com/pingcap-inc/labs.tidb.io/blob/4cf4a288439cb941dc2283ad1e8aafd479c510bd/frontend/src/lib/form.ts
 * @param formApi - The form API instance that contains field information and methods to apply errors.
 * @param body - The error body containing error messages keyed by field names.
 * @param phase - The validation phase under which the errors should be categorized.
 * @return Returns an object containing unhandled errors if any, or undefined if all errors are handled.
 */
function applyFormError<FormApi extends { fieldInfo: Record<string, FieldInfo<any>> }> (
  formApi: FormApi,
  body: Record<string, string[]>,
  phase: keyof ValidationErrorMap,
) {
  const unhandled: Record<string, string[]> = {};
  Object.entries(body).forEach(([key, value]) => {
    if (key in formApi.fieldInfo) {
      const field = formApi.fieldInfo[key]?.instance;
      if (field) {
        field.setErrorMap({ [phase]: value });
        return;
      }
    }
    unhandled[key] = value;
  });
  if (Object.keys(unhandled).length > 0) {
    return unhandled;
  } else {
    return undefined;
  }
}