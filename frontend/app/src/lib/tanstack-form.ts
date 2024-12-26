import type { DeepKeys, DeepValue, FormApi } from '@tanstack/react-form';

export function subscribeField<TData, TName extends DeepKeys<TData>> (
  form: FormApi<TData>,
  name: TName,
  cb: (value: DeepValue<TData, TName>, oldValue: DeepValue<TData, TName>) => void,
) {
  let oldValue = form.getFieldValue(name);

  return form.store.subscribe(() => {
    const newValue = form.getFieldValue(name);

    if (newValue !== oldValue) {
      const ol = oldValue;
      oldValue = newValue;
      cb(newValue, ol);
    }
  });
}
