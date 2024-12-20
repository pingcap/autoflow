'use client';

import { type CreateEmbeddingModel, createEmbeddingModel, type EmbeddingModel, listEmbeddingModelOptions, testEmbeddingModel } from '@/api/embedding-models';
import { ProviderSelect } from '@/components/form/biz';
import { FormInput } from '@/components/form/control-widget';
import { FormFieldBasicLayout } from '@/components/form/field-layout.beta';
import { FormRootErrorBeta as FormRootError } from '@/components/form/root-error';
import { onSubmitHelper } from '@/components/form/utils';
import { CodeInput } from '@/components/form/widgets/CodeInput';
import { ProviderDescription } from '@/components/provider-description';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Form, formDomEventHandlers, FormSubmit } from '@/components/ui/form.beta';
import { useModelProvider } from '@/hooks/use-model-provider';
import { zodJsonText } from '@/lib/zod';
import { useForm as useTanstackForm } from '@tanstack/react-form';
import { useId, useState } from 'react';
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
  const id = useId();
  const { data: options, isLoading, error } = useSWR('api.embedding-models.list-options', listEmbeddingModelOptions);
  const [submissionError, setSubmissionError] = useState<unknown>();

  const form = useTanstackForm<CreateEmbeddingModel | Omit<CreateEmbeddingModel, 'model' | 'credentials'>>({
    validators: {
      onSubmit: unsetForm,
    },
    onSubmit (props) {
      const { value } = props;
      const provider = options?.find(option => option.provider === value.provider);

      const schema = provider
        ? provider.credentials_type === 'str'
          ? strCredentialForm
          : provider.credentials_type === 'dict'
            ? dictCredentialForm
            : unsetForm
        : unsetForm;

      return onSubmitHelper(schema, async (values) => {
        const { error, success } = await testEmbeddingModel(values as CreateEmbeddingModel);
        if (!success) {
          throw new Error(error || 'Test Embedding Model failed');
        }
        const embeddingModel = await createEmbeddingModel(values as CreateEmbeddingModel);
        toast.success(`Embedding Model ${embeddingModel.name} successfully created.`);
        onCreated?.(embeddingModel);
      }, setSubmissionError)(props);
    },
    defaultValues: {
      name: '',
      provider: '',
      vector_dimension: 1536,
      config: '{}',
    },
  });

  const provider = useModelProvider(form, options, 'default_embedding_model');

  return (
    <>
      <Form form={form} disabled={transitioning} submissionError={submissionError}>
        <form id={id} className="space-y-4 max-w-screen-sm" {...formDomEventHandlers(form, transitioning)}>
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
          <FormRootError title="Failed to create Embedding Model" />
          <FormSubmit disabled={!options} transitioning={transitioning} form={id}>
            Create Embedding Model
          </FormSubmit>
        </form>
      </Form>
    </>
  );
}
