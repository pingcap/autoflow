'use client';

import { setDefault } from '@/api/commons';
import { deleteLlm, type LLM, updateLlm, type UpdateLLM } from '@/api/llms';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { ProviderSelect } from '@/components/form/biz';
import { FormInput, FormSwitch } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { ModelConnectionForm } from '@/components/model-connection-form';
import { useLLMProviders } from '@/components/llm/hooks';
import { fieldAccessor, GeneralSettingsField, type GeneralSettingsFieldAccessor, GeneralSettingsForm } from '@/components/settings-form';
import type { KeyOfType } from '@/lib/typing-utils';
import { format } from 'date-fns';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';
import { z } from 'zod';

export function UpdateLlmForm ({ llm }: { llm: LLM }) {
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();
  const { data: options, isLoading, error } = useLLMProviders();

  const provider = options?.find(option => option.provider === llm.provider);

  return (
    <div className="max-w-screen-sm space-y-4">
      <GeneralSettingsForm<UpdateLLM>
        data={llm}
        readonly={false}
        loading={transitioning}
        onUpdate={async (data, path) => {
          if (path[0] === 'is_default') {
            await setDefault('llms', llm.id);
          } else {
            const key = path[0] as keyof UpdateLLM;
            await updateLlm(llm.id, {
              [key]: data[key],
            });
          }
          startTransition(() => {
            router.refresh();
          });
        }}
      >
        <GeneralSettingsField accessor={idAccessor} schema={anySchema} readonly>
          <field.Basic label="ID" name="value">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={nameAccessor} schema={nameSchema}>
          <field.Basic label="Name" name="value">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={providerAccessor} schema={anySchema} readonly>
          <field.Basic label="Provider" name="value" description={provider?.provider_description}>
            <ProviderSelect options={options} isLoading={isLoading} error={error} />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={modelAccessor} schema={anySchema} readonly>
          <field.Basic label="Model" name="value" description={provider?.llm_model_description}>
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={isDefaultAccessor} schema={anySchema}>
          <field.Contained label="Is Default" name="value">
            <FormSwitch />
          </field.Contained>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={createdAtAccessor} schema={anySchema} readonly>
          <field.Basic label="Created At" name="value">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={updatedAtAccessor} schema={anySchema} readonly>
          <field.Basic label="Updated At" name="value">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
      </GeneralSettingsForm>
      <ModelConnectionForm
        config={llm.config}
        credentialType={provider?.credentials_type}
        disabled={transitioning}
        onSave={async value => {
          await updateLlm(llm.id, value);
          startTransition(() => { router.refresh(); });
        }}
      />
      <DangerousActionButton
        variant="destructive"
        disabled={transitioning}
        action={async () => {
          await deleteLlm(llm.id);
          startTransition(() => {
            router.push('/llms');
          });
        }}
      >
        Delete
      </DangerousActionButton>
    </div>
  );
}

const field = formFieldLayout<{ value: any | any[] }>();

const anySchema = z.any();

const getDatetimeAccessor = (key: KeyOfType<LLM, Date | undefined | null>): GeneralSettingsFieldAccessor<LLM, string> => {
  return {
    path: [key],
    get (data) {
      return format(data[key], 'yyyy-MM-dd HH:mm:ss');
    },
    set () {
      throw new Error(`update ${key} is not supported`);
    },
  };
};

const nameSchema = z.string();

const nameAccessor = fieldAccessor<UpdateLLM, 'name'>('name');
const idAccessor = fieldAccessor<LLM, 'id'>('id');
const providerAccessor = fieldAccessor<LLM, 'provider'>('provider');
const modelAccessor = fieldAccessor<LLM, 'model'>('model');

const isDefaultAccessor = fieldAccessor<LLM, 'is_default'>('is_default');
const createdAtAccessor = getDatetimeAccessor('created_at');
const updatedAtAccessor = getDatetimeAccessor('updated_at');
