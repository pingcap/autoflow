'use client';

import { type CreateLLM, createLlm, type LLM, testLlm } from '@/api/llms';
import { ProviderSelect } from '@/components/form/biz';
import { FormInput } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { FormRootError } from '@/components/form/root-error';
import { onSubmitHelper } from '@/components/form/utils';
import { CodeInput } from '@/components/form/widgets/CodeInput';
import { useLLMProviders } from '@/components/llm/hooks';
import { ProviderDescription } from '@/components/provider-description';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Form, formDomEventHandlers, FormSubmit } from '@/components/ui/form.beta';
import { useModelProvider } from '@/hooks/use-model-provider';
import { zodJsonText } from '@/lib/zod';
import { useForm } from '@tanstack/react-form';
import { useId, useState } from 'react';
import { toast } from 'sonner';
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

const field = formFieldLayout<CreateLLM>();

export function CreateLLMForm ({ transitioning, onCreated }: { transitioning?: boolean, onCreated?: (llm: LLM) => void }) {
  const id = useId();
  const { data: options, isLoading, error } = useLLMProviders();
  const [submissionError, setSubmissionError] = useState<unknown>();

  const form = useForm<CreateLLM | Omit<CreateLLM, 'model' | 'credentials'>>({
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
        const { error, success } = await testLlm(values as CreateLLM);
        if (!success) {
          throw new Error(error || 'Test LLM failed');
        }
        const llm = await createLlm(values as CreateLLM);
        toast.success(`LLM ${llm.name} successfully created.`);
        onCreated?.(llm);
      }, setSubmissionError)(props);
    },
    defaultValues: {
      name: '',
      provider: '',
      is_default: false,
      config: '{}',
    },
  });

  const provider = useModelProvider(form, options, 'default_llm_model');

  return (
    <>
      <Form form={form} disabled={transitioning} submissionError={submissionError}>
        <form id={id} className="space-y-4 max-w-screen-sm" {...formDomEventHandlers(form, transitioning)}>
          <field.Basic name="name" label="Name">
            <FormInput />
          </field.Basic>
          <field.Basic name="provider" label="Provider" description={provider && <ProviderDescription provider={provider} />}>
            <ProviderSelect options={options} isLoading={isLoading} error={error} />
          </field.Basic>
          {provider && (
            <>
              <field.Basic name="model" label="Model" description={provider.llm_model_description}>
                <FormInput />
              </field.Basic>
              <field.Basic name="credentials" label={provider.credentials_display_name} description={provider.credentials_description}>
                {provider.credentials_type === 'str'
                  ? <FormInput placeholder={provider.default_credentials} />
                  : <CodeInput language="json" placeholder={JSON.stringify(provider.default_credentials, undefined, 2)} />
                }
              </field.Basic>
              <Accordion type="multiple">
                <AccordionItem value="advanced-settings">
                  <AccordionTrigger>
                    Advanced Settings
                  </AccordionTrigger>
                  <AccordionContent className="px-4">
                    <field.Basic name="config" label="Config" description={provider.config_description}>
                      <CodeInput language="json" />
                    </field.Basic>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </>
          )}
          <FormRootError title="Failed to create LLM" />
          <FormSubmit disabled={!options} transitioning={transitioning} form={id}>
            Create LLM
          </FormSubmit>
        </form>
      </Form>
    </>
  );
}
