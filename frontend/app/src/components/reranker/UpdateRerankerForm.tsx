'use client';

import { setDefault } from '@/api/commons';
import { deleteReranker, type Reranker, updateReranker, type UpdateReranker } from '@/api/rerankers';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { ProviderSelect } from '@/components/form/biz';
import { FormInput, FormSwitch } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { ModelConnectionForm } from '@/components/model-connection-form';
import { useRerankerProviders } from '@/components/reranker/hooks';
import { fieldAccessor, GeneralSettingsField, type GeneralSettingsFieldAccessor, GeneralSettingsForm } from '@/components/settings-form';
import type { KeyOfType } from '@/lib/typing-utils';
import { format } from 'date-fns';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';
import { z } from 'zod';

export function UpdateRerankerForm ({ reranker }: { reranker: Reranker }) {
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();
  const { data: options, isLoading, error } = useRerankerProviders();

  const provider = options?.find(option => option.provider === reranker.provider);

  return (
    <div className="max-w-screen-sm space-y-4">
      <GeneralSettingsForm<UpdateReranker>
        data={reranker}
        readonly={false}
        loading={transitioning}
        onUpdate={async (data, path) => {
          if (path[0] === 'is_default') {
            await setDefault('embedding-models', reranker.id);
          } else {
            const key = path[0] as keyof UpdateReranker;
            await updateReranker(reranker.id, {
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
          <field.Basic label="Model" name="value" description={provider?.reranker_model_description}>
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={topNAccessor} schema={topNSchema}>
          <field.Basic label="Top N" name="value">
            <FormInput type="number" min={1} step={1} />
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
        config={reranker.config}
        credentialType={provider?.credentials_type}
        disabled={transitioning}
        onSave={async value => {
          await updateReranker(reranker.id, value);
          startTransition(() => { router.refresh(); });
        }}
      />
      <DangerousActionButton
        variant="destructive"
        disabled={transitioning}
        action={async () => {
          await deleteReranker(reranker.id);
          startTransition(() => {
            router.push('/reranker-models');
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

const getDatetimeAccessor = (key: KeyOfType<Reranker, Date | undefined | null>): GeneralSettingsFieldAccessor<Reranker, string> => {
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
const topNSchema = z.coerce.number().int().min(1);

const nameAccessor = fieldAccessor<UpdateReranker, 'name'>('name');
const idAccessor = fieldAccessor<Reranker, 'id'>('id');
const providerAccessor = fieldAccessor<Reranker, 'provider'>('provider');
const modelAccessor = fieldAccessor<Reranker, 'model'>('model');
const topNAccessor = fieldAccessor<Reranker, 'top_n'>('top_n');

const isDefaultAccessor = fieldAccessor<Reranker, 'is_default'>('is_default');
const createdAtAccessor = getDatetimeAccessor('created_at');
const updatedAtAccessor = getDatetimeAccessor('updated_at');
