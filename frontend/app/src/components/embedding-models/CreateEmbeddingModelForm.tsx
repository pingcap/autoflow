'use client';

import { createEmbeddingModel, type EmbeddingModel, listEmbeddingModelOptions, testEmbeddingModel } from '@/api/embedding-models';
import { ProviderSelect } from '@/components/form/biz';
import { FormInput, FormSwitch } from '@/components/form/control-widget';
import { FormFieldBasicLayout, FormFieldContainedLayout } from '@/components/form/field-layout';
import { FormRootError } from '@/components/form/root-error';
import { CodeInput } from '@/components/form/widgets/CodeInput';
import { ProviderDescription } from '@/components/provider-description';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { getErrorMessage } from '@/lib/errors';
import { zodJsonText } from '@/lib/zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Loader2Icon } from 'lucide-react';
import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import useSWR from 'swr';
import { z } from 'zod';

const unsetForm = z.object({
  name: z.string().min(1, 'Must not empty'),
  provider: z.string().min(1, 'Must not empty'),
  vector_dimension: z.coerce.number().int().positive(),
  config: zodJsonText().optional(),
});

const strCredentialForm = unsetForm.extend({
  model: z.string().min(1, 'Must not empty'),
  credentials: z.string().min(1, 'Must not empty'),
});

const dictCredentialForm = unsetForm.extend({
  model: z.string().min(1, 'Must not empty'),
  credentials: z.object({}).passthrough(),
});

export function CreateEmbeddingModelForm ({ transitioning, onCreated }: { transitioning?: boolean, onCreated?: (embeddingModel: EmbeddingModel) => void }) {
  const { data: options, isLoading, error } = useSWR('api.embedding-models.list-options', listEmbeddingModelOptions);

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
    },
  });

  const providerName = form.watch('provider');
  const provider = options?.find(option => option.provider === providerName);

  useEffect(() => {
    if (provider) {
      form.reset({
        ...form.getValues(),
        model: provider.default_embedding_model,
        credentials: provider.credentials_type === 'dict' ? undefined : '',
        config: JSON.stringify(provider.default_config, undefined, 2),
      });
    } else {
      const { name } = form.getValues();
      form.reset({
        name,
        provider: '',
        credentials: '',
        model: '',
        config: '{}',
      });
    }
  }, [provider]);

  const handleSubmit = form.handleSubmit(async (values) => {
    const { error, success } = await testEmbeddingModel(values);
    if (!success) {
      form.setError('root', { message: error || 'Unknown error' });
      return;
    }
    try {
      const embeddingModel = await createEmbeddingModel(values);
      toast('Embedding Model successfully created.');
      onCreated?.(embeddingModel);
    } catch (error) {
      form.setError('root', { message: getErrorMessage(error) });
    }
  });

  return (
    <>
      <Form {...form}>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <FormFieldBasicLayout name="name" label="Name">
            <FormInput />
          </FormFieldBasicLayout>
          <FormFieldBasicLayout name="provider" label="Provider" description={provider && <ProviderDescription provider={provider} />}>
            <ProviderSelect options={options} isLoading={isLoading} error={error} />
          </FormFieldBasicLayout>
          {provider && (
            <>
              <FormFieldBasicLayout name="model" label="Model" description={provider.embedding_model_description}>
                <FormInput />
              </FormFieldBasicLayout>
              <FormFieldBasicLayout name="credentials" label={provider.credentials_display_name} description={provider.credentials_description}>
                {provider.credentials_type === 'str'
                  ? <FormInput placeholder={provider.default_credentials} />
                  : <CodeInput language="json" placeholder={JSON.stringify(provider.default_credentials, undefined, 2)} />
                }
              </FormFieldBasicLayout>
              <FormFieldBasicLayout name="vector_dimension" label="Vector Dimensions">
                <FormInput type="number" min={1} />
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
          <FormFieldContainedLayout name="is_default" label="Is Default Embedding Model" description="Enable will unset original default Embedding Model.">
            <FormSwitch />
          </FormFieldContainedLayout>
          <FormRootError title="Failed to create LLM" />
          <FormRootError title="Failed to create Embedding Model" />
          <Button disabled={form.formState.disabled || !provider || form.formState.isSubmitting || transitioning}>
            {(form.formState.isSubmitting || transitioning) && <Loader2Icon className="size-4 mr-1 animate-spin repeat-infinite" />}
            Create Embedding Model
          </Button>
        </form>
      </Form>
    </>
  );
}
