import type { ProviderOption } from '@/api/providers';
import { subscribeField } from '@/lib/tanstack-form';
import type { KeyOfType } from '@/lib/typing-utils';
import type { FormApi, ReactFormExtendedApi } from '@tanstack/react-form';
import { useEffect, useState } from 'react';

export function useModelProvider<O extends ProviderOption, F extends { provider: string, model?: string, credentials?: string | object, config?: any }> (
  propForm: ReactFormExtendedApi<F>,
  options: O[] | undefined,
  defaultModelField: KeyOfType<O, string>,
) {
  const form = propForm as never as FormApi<{ provider: string, model?: string, credentials?: string, config: string }>;

  const [provider, setProvider] = useState<O | undefined>(() => options?.find(option => option.provider === form.getFieldValue('provider' as never)));

  useEffect(() => {
    let lastProvider = form.getFieldValue('provider' as never);
    const o = options?.find(option => option.provider === lastProvider);
    setProvider(o);

    return subscribeField(form, 'provider', name => {
      const provider = options?.find(option => option.provider === name);

      if (name) {
        setProvider(provider);
      } else {
        setProvider(undefined);
      }

      if (provider) {
        form.store.batch(() => {
          form.setFieldValue('model', provider[defaultModelField] as string);
          form.setFieldValue('credentials', '');
          form.setFieldValue('config', JSON.stringify(provider.default_config, undefined, 2));
        });
      } else if (name) {
        // Provider not found, clear all provider spec data.
        form.store.batch(() => {
          if (name) {
            form.fieldInfo.provider?.instance?.setErrorMap({
              onChange: `Invalid provider '${name}'`,
            });
          }
          form.setFieldValue('model', '');
          form.setFieldValue('credentials', '');
          form.setFieldValue('config', '{}');
        });
      }
    });
  }, [form, options, defaultModelField]);

  return provider;
}
