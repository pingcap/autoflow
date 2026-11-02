'use client';

import { setDefault } from '@/api/commons';
import { type EmbeddingModel, updateEmbeddingModel, type UpdateEmbeddingModel } from '@/api/embedding-models';
import { useEmbeddingModelProviders } from '@/components/embedding-models/hooks';
import { ProviderSelect } from '@/components/form/biz';
import { FormInput, FormSwitch } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { ModelConnectionForm } from '@/components/model-connection-form';
import { fieldAccessor, GeneralSettingsField, type GeneralSettingsFieldAccessor, GeneralSettingsForm } from '@/components/settings-form';
import type { KeyOfType } from '@/lib/typing-utils';
import { format } from 'date-fns';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';
import { z } from 'zod';

export function UpdateEmbeddingModelForm ({ embeddingModel }: { embeddingModel: EmbeddingModel }) {
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();
  const { data: options, isLoading, error } = useEmbeddingModelProviders();

  const provider = options?.find(option => option.provider === embeddingModel.provider);

  return (
    <div className="max-w-screen-sm space-y-4">
      <GeneralSettingsForm<UpdateEmbeddingModel>
        data={embeddingModel}
        readonly={false}
        loading={transitioning}
        onUpdate={async (data, path) => {
          if (path[0] === 'is_default') {
            await setDefault('embedding-models', embeddingModel.id);
          } else {
            const key = path[0] as keyof UpdateEmbeddingModel;
            await updateEmbeddingModel(embeddingModel.id, {
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
          <field.Basic label="Model" name="value" description={provider?.embedding_model_description}>
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={vectorDimensionAccessor} schema={anySchema} readonly>
          <field.Basic label="Vector Dimensions" name="value">
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
        config={embeddingModel.config}
        credentialType={provider?.credentials_type}
        disabled={transitioning}
        onSave={async value => {
          await updateEmbeddingModel(embeddingModel.id, value);
          startTransition(() => { router.refresh(); });
        }}
      />
    </div>
  );
}

const field = formFieldLayout<{ value: any | any[] }>();

const anySchema = z.any();

const getDatetimeAccessor = (key: KeyOfType<EmbeddingModel, Date | undefined | null>): GeneralSettingsFieldAccessor<EmbeddingModel, string> => {
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

const nameAccessor = fieldAccessor<UpdateEmbeddingModel, 'name'>('name');
const idAccessor = fieldAccessor<EmbeddingModel, 'id'>('id');
const providerAccessor = fieldAccessor<EmbeddingModel, 'provider'>('provider');
const modelAccessor = fieldAccessor<EmbeddingModel, 'model'>('model');
const vectorDimensionAccessor = fieldAccessor<EmbeddingModel, 'vector_dimension'>('vector_dimension');

const isDefaultAccessor = fieldAccessor<EmbeddingModel, 'is_default'>('is_default');
const createdAtAccessor = getDatetimeAccessor('created_at');
const updatedAtAccessor = getDatetimeAccessor('updated_at');
