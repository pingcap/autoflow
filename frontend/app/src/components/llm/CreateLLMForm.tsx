'use client';

import { createLlm, listLlmOptions, type LLM, testLlm } from '@/api/llms';
import { ProviderSelect } from '@/components/form/biz';
import { FormInput } from '@/components/form/control-widget';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import { FormRootError } from '@/components/form/root-error';
import { handleSubmitHelper } from '@/components/form/utils';
import { CodeInput } from '@/components/form/widgets/CodeInput';
import { ProviderDescription } from '@/components/provider-description';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { zodJsonText } from '@/lib/zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Loader2Icon } from 'lucide-react';
import { useEffect, useId } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import useSWR from 'swr';
import { z } from 'zod';

const unsetForm = z.object({
  name: z.string().min(1, 'Must not empty'),
  provider: z.string().min(1, 'Must not empty'),
  config: zodJsonText().optional(),
  is_default: z.boolean().optional(),
});

const strCredentialForm = unsetForm.extend({
  model: z.string().min(1, 'Must not empty'),
  credentials: z.string().min(1, 'Must not empty'),
});

const dictCredentialForm = unsetForm.extend({
  model: z.string().min(1, 'Must not empty'),
  credentials: zodJsonText(),
});

export function CreateLLMForm ({ transitioning, onCreated }: { transitioning?: boolean, onCreated?: (llm: LLM) => void }) {
  const id = useId();
  const { data: options, isLoading, error } = useSWR('api.llms.list-options', listLlmOptions);

  const form = useForm<any>({
    resolver: (values, context, opts) => {
      const provider = options?.find(option => option.provider === values.provider);

      return zodResolver(
        provider
          ? provider.credentials_type === 'str'
            ? strCredentialForm
            : provider.credentials_type === 'dict'
              ? dictCredentialForm
              : unsetForm
          : unsetForm)(values, context, opts);
    },
    defaultValues: {
      name: '',
      provider: '',
      model: '',
      credentials: '',
      is_default: false,
      config: '{}',
    },
  });

  const providerName = form.watch('provider');
  const provider = options?.find(option => option.provider === providerName);

  useEffect(() => {
    if (provider) {
      form.reset({
        ...form.getValues(),
        model: provider.default_llm_model,
        credentials: provider.credentials_type === 'dict' ? undefined : '',
        config: JSON.stringify(provider.default_config, undefined, 2),
      });
    } else {
      const { name, is_default } = form.getValues();
      form.reset({
        name,
        is_default,
        provider: '',
        credentials: '',
        model: '',
        config: '{}',
      });
    }
  }, [provider]);

  const handleSubmit = handleSubmitHelper(form, async (values) => {
    const { error, success } = await testLlm(values);
    if (!success) {
      throw new Error(error || 'Test LLM failed');
    }
    const llm = await createLlm(values);
    toast.success(`LLM ${llm.name} successfully created.`);
    onCreated?.(llm);
  });

  return (
    <>
      <Form {...form}>
        <form id={id} className="space-y-4" onSubmit={handleSubmit}>
          <FormFieldBasicLayout name="name" label="Name">
            <FormInput />
          </FormFieldBasicLayout>
          <FormFieldBasicLayout name="provider" label="Provider" description={provider && <ProviderDescription provider={provider} />}>
            <ProviderSelect options={options} isLoading={isLoading} error={error} />
          </FormFieldBasicLayout>
          {provider && (
            <>
              <FormFieldBasicLayout name="model" label="Model" description={provider.llm_model_description}>
                <FormInput />
              </FormFieldBasicLayout>
              <FormFieldBasicLayout name="credentials" label={provider.credentials_display_name} description={provider.credentials_description}>
                {provider.credentials_type === 'str'
                  ? <FormInput placeholder={provider.default_credentials} />
                  : <CodeInput language="json" placeholder={JSON.stringify(provider.default_credentials, undefined, 2)} />
                }
              </FormFieldBasicLayout>
              <Accordion type="multiple">
                <AccordionItem value="advanced-settings">
                  <AccordionTrigger>
                    Advanced Settings
                  </AccordionTrigger>
                  <AccordionContent className="px-4">
                    <FormFieldBasicLayout name="config" label="Config" description={provider.config_description}>
                      <CodeInput language="json" />
                    </FormFieldBasicLayout>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </>
          )}
          <FormRootError title="Failed to create LLM" />
          <Button disabled={form.formState.disabled || !provider || form.formState.isSubmitting || transitioning} type="submit" form={id}>
            {(form.formState.isSubmitting || transitioning) && <Loader2Icon className="size-4 mr-1 animate-spin repeat-infinite" />}
            Create LLM
          </Button>
        </form>
      </Form>
    </>
  );
}
