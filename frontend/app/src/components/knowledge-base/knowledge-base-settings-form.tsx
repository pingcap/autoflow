'use client';

import { type KnowledgeBase, type KnowledgeBaseIndexMethod, updateKnowledgeBase } from '@/api/knowledge-base';
import { EmbeddingModelSelect, LLMSelect } from '@/components/form/biz';
import { FormInput, FormSwitch, FormTextarea } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { mutateKnowledgeBases } from '@/components/knowledge-base/hooks';
import { fieldAccessor, type GeneralSettingsFieldAccessor, GeneralSettingsForm, shallowPick } from '@/components/settings-form';
import { GeneralSettingsField as GeneralSettingsField } from '@/components/settings-form/GeneralSettingsField';
import type { KeyOfType } from '@/lib/typing-utils';
import { format } from 'date-fns';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';
import { z } from 'zod';

const field = formFieldLayout<{ value: any }>();

export function KnowledgeBaseSettingsForm ({ knowledgeBase }: { knowledgeBase: KnowledgeBase }) {
  const router = useRouter();
  const [transitioning, startTransition] = useTransition();

  return (
    <GeneralSettingsForm
      data={knowledgeBase}
      readonly={false}
      loading={transitioning}
      onUpdate={async (data, path) => {
        if (['name', 'description'].includes(path[0] as never)) {
          const partial = shallowPick(data, path as never);
          await updateKnowledgeBase(knowledgeBase.id, partial);
          startTransition(() => {
            router.refresh();
            mutateKnowledgeBases();
          });
        } else {
          throw new Error(`${path.map(p => String(p)).join('.')} is not updatable currently.`);
        }
      }}>
      <GeneralSettingsField schema={nameSchema} accessor={nameAccessor}>
        <field.Basic name="value" label="Name">
          <FormInput />
        </field.Basic>
      </GeneralSettingsField>
      <GeneralSettingsField schema={descriptionSchema} accessor={descriptionAccessor}>
        <field.Basic name="value" label="Description">
          <FormTextarea />
        </field.Basic>
      </GeneralSettingsField>
      <GeneralSettingsField readonly schema={llmSchema} accessor={llmAccessor}>
        <field.Basic name="value" label="LLM">
          <LLMSelect />
        </field.Basic>
      </GeneralSettingsField>
      <GeneralSettingsField readonly schema={embeddingModelSchema} accessor={embeddingModelAccessor}>
        <field.Basic name="value" label="Embedding Model">
          <EmbeddingModelSelect />
        </field.Basic>
      </GeneralSettingsField>
      <GeneralSettingsField readonly accessor={vectorAccessor} schema={vectorSchema}>
        <field.Contained name="value" label="Vector Index" description="/// TBD">
          <FormSwitch />
        </field.Contained>
      </GeneralSettingsField>
      <GeneralSettingsField readonly accessor={kgAccessor} schema={kgSchema}>
        <field.Contained name="value" label="Knowledge Graph Index" description="/// TBD">
          <FormSwitch />
        </field.Contained>
      </GeneralSettingsField>
      <GeneralSettingsField readonly schema={createdAtSchema} accessor={createdAtAccessor}>
        <field.Basic name="value" label="Created At">
          <FormInput />
        </field.Basic>
      </GeneralSettingsField>
      <GeneralSettingsField readonly schema={updatedAtSchema} accessor={updatedAtAccessor}>
        <field.Basic name="value" label="Updated At">
          <FormInput />
        </field.Basic>
      </GeneralSettingsField>
    </GeneralSettingsForm>
  );
}

const getIndexMethodAccessor = (method: KnowledgeBaseIndexMethod): GeneralSettingsFieldAccessor<KnowledgeBase, boolean> => ({
  path: ['index_methods'],
  get: data => data.index_methods.includes('vector'),
  set: (data, value) => {
    if (value) {
      return {
        ...data,
        index_methods: Array.from(new Set(data.index_methods.concat(method))),
      };
    } else {
      return {
        ...data,
        index_methods: data.index_methods.filter(m => m !== method),
      };
    }
  },
});
const getDatetimeAccessor = (key: KeyOfType<KnowledgeBase, Date>): GeneralSettingsFieldAccessor<KnowledgeBase, string> => {
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
const nameAccessor = fieldAccessor<KnowledgeBase, 'name'>('name');

const descriptionSchema = z.string();
const descriptionAccessor = fieldAccessor<KnowledgeBase, 'description'>('description');

const vectorSchema = z.boolean();
const vectorAccessor = getIndexMethodAccessor('vector');

const kgSchema = z.boolean();
const kgAccessor = getIndexMethodAccessor('knowledge_graph');

const llmSchema = z.number();
const llmAccessor: GeneralSettingsFieldAccessor<KnowledgeBase, number | undefined> = {
  path: ['llm'],
  get (data) {
    return data.llm?.id;
  },
  set () {
    throw new Error('TODO');
  },
};

const embeddingModelSchema = z.number();
const embeddingModelAccessor: GeneralSettingsFieldAccessor<KnowledgeBase, number | undefined> = {
  path: ['embedding_model'],
  get (data) {
    return data.embedding_model?.id;
  },
  set () {
    throw new Error('TODO');
  },
};

const createdAtSchema = z.string();
const createdAtAccessor = getDatetimeAccessor('created_at');

const updatedAtSchema = z.string();
const updatedAtAccessor = getDatetimeAccessor('updated_at');
