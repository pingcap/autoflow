'use client';

import { type ChatEngine, type ChatEngineKnowledgeGraphOptions, type ChatEngineLLMOptions, type ChatEngineOptions, updateChatEngine } from '@/api/chat-engines';
import { KBListSelect } from '@/components/chat-engine/kb-list-select';
import { LLMSelect, RerankerSelect } from '@/components/form/biz';
import { FormCheckbox, FormInput, FormSwitch } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { PromptInput } from '@/components/form/widgets/PromptInput';
import { SecondaryNavigatorItem, SecondaryNavigatorLayout, SecondaryNavigatorList, SecondaryNavigatorMain } from '@/components/secondary-navigator-list';
import { fieldAccessor, GeneralSettingsField as GeneralSettingsField, type GeneralSettingsFieldAccessor, GeneralSettingsForm, shallowPick } from '@/components/settings-form';
import type { KeyOfType } from '@/lib/typing-utils';
import { capitalCase } from 'change-case-all';
import { format } from 'date-fns';
import { useRouter } from 'next/navigation';
import { type ReactNode, useTransition } from 'react';
import { z } from 'zod';

const field = formFieldLayout<{ value: any | any[] }>();

export function UpdateChatEngineForm ({ chatEngine, defaultChatEngineOptions }: { chatEngine: ChatEngine, defaultChatEngineOptions: ChatEngineOptions }) {
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();

  return (
    <GeneralSettingsForm
      data={chatEngine}
      readonly={false}
      loading={transitioning}
      onUpdate={async (data, path) => {
        if (updatableFields.includes(path[0] as any)) {
          const partial = shallowPick(data, path as [(typeof updatableFields)[number], ...any[]]);
          await updateChatEngine(chatEngine.id, partial);
          startTransition(() => {
            router.refresh();
          });
        } else {
          throw new Error(`${path.map(p => String(p)).join('.')} is not updatable currently.`);
        }
      }}
    >
      <SecondaryNavigatorLayout defaultValue="Info">
        <SecondaryNavigatorList>
          <SecondaryNavigatorItem value="Info">
            Info
          </SecondaryNavigatorItem>
          <SecondaryNavigatorItem value="Retrieval">
            Retrieval
          </SecondaryNavigatorItem>
          <SecondaryNavigatorItem value="Generation">
            Generation
          </SecondaryNavigatorItem>
          <SecondaryNavigatorItem value="Features">
            Features
          </SecondaryNavigatorItem>
        </SecondaryNavigatorList>
        <Section title="Info">
          <GeneralSettingsField readonly accessor={idAccessor} schema={neverSchema}>
            <field.Basic required name="value" label="ID">
              <FormInput />
            </field.Basic>
          </GeneralSettingsField>
          <GeneralSettingsField accessor={nameAccessor} schema={nameSchema}>
            <field.Basic name="value" label="Name">
              <FormInput />
            </field.Basic>
          </GeneralSettingsField>
          <GeneralSettingsField accessor={isDefaultAccessor} schema={isDefaultSchema}>
            <field.Contained unimportant name="value" label="Is Default" fallbackValue={chatEngine.is_default} description="/// Description TBD">
              <FormSwitch />
            </field.Contained>
          </GeneralSettingsField>
          <GeneralSettingsField readonly accessor={createdAccessor} schema={neverSchema}>
            <field.Basic name="value" label="Created At">
              <FormInput />
            </field.Basic>
          </GeneralSettingsField>
          <GeneralSettingsField readonly accessor={updatedAccessor} schema={neverSchema}>
            <field.Basic name="value" label="Updated At">
              <FormInput />
            </field.Basic>
          </GeneralSettingsField>
          <SubSection title="Models">
            <GeneralSettingsField accessor={llmIdAccessor} schema={idSchema}>
              <field.Basic name="value" label="LLM">
                <LLMSelect />
              </field.Basic>
            </GeneralSettingsField>
            <GeneralSettingsField accessor={fastLlmIdAccessor} schema={idSchema}>
              <field.Basic name="value" label="Fast LLM">
                <LLMSelect />
              </field.Basic>
            </GeneralSettingsField>
          </SubSection>
          <SubSection title="External Engine Config">
            <GeneralSettingsField accessor={externalEngineAccessor} schema={externalEngineSchema}>
              <field.Basic name="value" label="External Chat Engine API URL (StackVM)" fallbackValue={defaultChatEngineOptions.external_engine_config?.stream_chat_api_url ?? ''}>
                <FormInput />
              </field.Basic>
            </GeneralSettingsField>
            <GeneralSettingsField accessor={llmAccessor.generate_goal_prompt} schema={llmSchema}>
              <field.Basic name="value" label="Generate Goal Prompt" description="/// TBD" fallbackValue={defaultChatEngineOptions.llm?.generate_goal_prompt}>
                <PromptInput />
              </field.Basic>
            </GeneralSettingsField>
          </SubSection>
        </Section>

        <Section title="Retrieval">
          <GeneralSettingsField accessor={kbAccessor} schema={kbSchema}>
            <field.Basic required name="value" label="Linked Knowledge Bases">
              <KBListSelect />
            </field.Basic>
          </GeneralSettingsField>
          <GeneralSettingsField accessor={rerankerIdAccessor} schema={idSchema}>
            <field.Basic name="value" label="Reranker">
              <RerankerSelect />
            </field.Basic>
          </GeneralSettingsField>
          <SubSection title="Knowledge Graph">
            <GeneralSettingsField accessor={kgEnabledAccessor} schema={kgEnabledSchema}>
              <field.Contained name="value" label="Enable Knowledge Graph" fallbackValue={defaultChatEngineOptions.knowledge_graph?.enabled} description="/// Description TBD">
                <FormSwitch />
              </field.Contained>
            </GeneralSettingsField>
            <GeneralSettingsField accessor={kgDepthAccessor} schema={kgDepthSchema}>
              <field.Basic name="value" label="Depth" fallbackValue={defaultChatEngineOptions.knowledge_graph?.depth}>
                <FormInput type="number" min={1} />
              </field.Basic>
            </GeneralSettingsField>
            <GeneralSettingsField accessor={kgIncludeMetaAccessor} schema={kgIncludeMetaSchema}>
              <field.Inline name="value" label="Include Meta" fallbackValue={defaultChatEngineOptions.knowledge_graph?.include_meta} description="/// Description TBD">
                <FormCheckbox />
              </field.Inline>
            </GeneralSettingsField>
            <GeneralSettingsField accessor={kgWithDegreeAccessor} schema={kgWithDegreeSchema}>
              <field.Inline name="value" label="With Degree" fallbackValue={defaultChatEngineOptions.knowledge_graph?.with_degree} description="/// Description TBD">
                <FormCheckbox />
              </field.Inline>
            </GeneralSettingsField>
            <GeneralSettingsField accessor={kgUsingIntentSearchAccessor} schema={kgUsingIntentSearchSchema}>
              <field.Inline name="value" label="Using intent search" fallbackValue={defaultChatEngineOptions.knowledge_graph?.using_intent_search} description="/// Description TBD">
                <FormCheckbox />
              </field.Inline>
            </GeneralSettingsField>
            {(['intent_graph_knowledge', 'normal_graph_knowledge'] as const).map(type => (
              <GeneralSettingsField key={type} accessor={llmAccessor[type]} schema={llmSchema}>
                <field.Basic name="value" label={capitalCase(type)} description="/// TBD" fallbackValue={defaultChatEngineOptions.llm?.[type]}>
                  <PromptInput />
                </field.Basic>
              </GeneralSettingsField>
            ))}
          </SubSection>
        </Section>

        <Section title="Generation">
          {(['condense_question_prompt', 'condense_answer_prompt', 'text_qa_prompt', 'refine_prompt'] as const).map(type => (
            <GeneralSettingsField key={type} accessor={llmAccessor[type]} schema={llmSchema}>
              <field.Basic name="value" label={capitalCase(type)} description="/// TBD" fallbackValue={defaultChatEngineOptions.llm?.[type]}>
                <PromptInput />
              </field.Basic>
            </GeneralSettingsField>
          ))}
        </Section>

        <Section title="Features">
          <GeneralSettingsField accessor={hideSourcesAccessor} schema={hideSourcesSchema}>
            <field.Inline name="value" label="Hide Sources" fallbackValue={defaultChatEngineOptions.hide_sources} description="/// Description TBD">
              <FormCheckbox />
            </field.Inline>
          </GeneralSettingsField>
          <SubSection title="Clarify Question">
            <GeneralSettingsField accessor={clarifyAccessor} schema={clarifyAccessorSchema}>
              <field.Contained unimportant name="value" label="Clarify Question" fallbackValue={defaultChatEngineOptions.clarify_question} description="/// Description TBD">
                <FormSwitch />
              </field.Contained>
            </GeneralSettingsField>
            <GeneralSettingsField accessor={llmAccessor.clarifying_question_prompt} schema={llmSchema}>
              <field.Contained name="value" label="Clarifying Question Prompt" description="/// TBD" fallbackValue={defaultChatEngineOptions.llm?.clarifying_question_prompt}>
                <PromptInput />
              </field.Contained>
            </GeneralSettingsField>
          </SubSection>
          <SubSection title="Post Verification">
            <GeneralSettingsField accessor={postVerificationUrlAccessor} schema={postVerificationUrlSchema}>
              <field.Basic name="value" label="Post Verifycation Service URL" fallbackValue={defaultChatEngineOptions.post_verification_url ?? ''}>
                <FormInput />
              </field.Basic>
            </GeneralSettingsField>
            <GeneralSettingsField accessor={postVerificationTokenAccessor} schema={postVerificationTokenSchema}>
              <field.Basic name="value" label="Post Verifycation Service Token" fallbackValue={defaultChatEngineOptions.post_verification_token ?? ''}>
                <FormInput />
              </field.Basic>
            </GeneralSettingsField>
          </SubSection>
          <SubSection title="Further Recommended Questions">
            <GeneralSettingsField accessor={llmAccessor.further_questions_prompt} schema={llmSchema}>
              <field.Basic name="value" label="Further Questions Prompt" description="/// TBD" fallbackValue={defaultChatEngineOptions.llm?.further_questions_prompt}>
                <PromptInput />
              </field.Basic>
            </GeneralSettingsField>
          </SubSection>
        </Section>
      </SecondaryNavigatorLayout>
    </GeneralSettingsForm>
  );
}

const updatableFields = ['name', 'llm_id', 'fast_llm_id', 'reranker_id', 'engine_options', 'is_default'] as const;

function optionAccessor<K extends keyof ChatEngineOptions> (key: K): GeneralSettingsFieldAccessor<ChatEngine, ChatEngineOptions[K]> {
  return {
    path: ['engine_options', key],
    get (engine) {
      return engine.engine_options[key];
    },
    set (engine, value) {
      return {
        ...engine,
        engine_options: {
          ...engine.engine_options,
          [key]: value,
        },
      };
    },
  };
}

function kgOptionAccessor<K extends keyof ChatEngineKnowledgeGraphOptions> (key: K): GeneralSettingsFieldAccessor<ChatEngine, ChatEngineKnowledgeGraphOptions[K]> {
  return {
    path: ['engine_options', 'knowledge_graph', key],
    get (engine) {
      return engine.engine_options.knowledge_graph?.[key];
    },
    set (engine, value) {
      return {
        ...engine,
        engine_options: {
          ...engine.engine_options,
          knowledge_graph: {
            ...engine.engine_options.knowledge_graph,
            [key]: value,
          },
        },
      };
    },
  };
}

function llmOptionAccessor<K extends keyof ChatEngineLLMOptions> (key: K): GeneralSettingsFieldAccessor<ChatEngine, ChatEngineLLMOptions[K]> {
  return {
    path: ['engine_options', 'llm', key],
    get (engine) {
      return engine.engine_options.llm?.[key];
    },
    set (engine, value) {
      return {
        ...engine,
        engine_options: {
          ...engine.engine_options,
          llm: {
            ...engine.engine_options.llm,
            [key]: value,
          },
        },
      };
    },
  };
}

const getDatetimeAccessor = (key: KeyOfType<ChatEngine, Date>): GeneralSettingsFieldAccessor<ChatEngine, string> => {
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

const idAccessor = fieldAccessor<ChatEngine, 'id'>('id');

const createdAccessor = getDatetimeAccessor('created_at');
const updatedAccessor = getDatetimeAccessor('updated_at');
const neverSchema = z.never();

const nameAccessor = fieldAccessor<ChatEngine, 'name'>('name');
const nameSchema = z.string().min(1);

const clarifyAccessor = optionAccessor('clarify_question');
const clarifyAccessorSchema = z.boolean().nullable().optional();

const isDefaultAccessor = fieldAccessor<ChatEngine, 'is_default'>('is_default');
const isDefaultSchema = z.boolean();

const getIdAccessor = (id: KeyOfType<ChatEngine, number | null>) => fieldAccessor<ChatEngine, KeyOfType<ChatEngine, number | null>>(id);
const idSchema = z.number().nullable();
const llmIdAccessor = getIdAccessor('llm_id');
const fastLlmIdAccessor = getIdAccessor('fast_llm_id');
const rerankerIdAccessor = getIdAccessor('reranker_id');

const kbAccessor: GeneralSettingsFieldAccessor<ChatEngine, number[] | null> = {
  path: ['engine_options'],
  get (data) {
    console.log(data.engine_options.knowledge_base?.linked_knowledge_bases?.map(kb => kb.id) ?? null);
    return data.engine_options.knowledge_base?.linked_knowledge_bases?.map(kb => kb.id) ?? null;
  },
  set (data, value) {
    return {
      ...data,
      engine_options: {
        ...data.engine_options,
        knowledge_base: {
          linked_knowledge_base: undefined,
          linked_knowledge_bases: value?.map(id => ({ id })) ?? null,
        },
      },
    };
  },
};
const kbSchema = z.number().array().min(1);

const kgEnabledAccessor = kgOptionAccessor('enabled');
const kgEnabledSchema = z.boolean().nullable();

const kgWithDegreeAccessor = kgOptionAccessor('with_degree');
const kgWithDegreeSchema = z.boolean().nullable();

const kgIncludeMetaAccessor = kgOptionAccessor('include_meta');
const kgIncludeMetaSchema = z.boolean().nullable();

const kgUsingIntentSearchAccessor = kgOptionAccessor('using_intent_search');
const kgUsingIntentSearchSchema = z.boolean().nullable();

const kgDepthAccessor = kgOptionAccessor('depth');
const kgDepthSchema = z.number().int().min(1).nullable();

const hideSourcesAccessor = optionAccessor('hide_sources');
const hideSourcesSchema = z.boolean().nullable();

const llmPromptFields = [
  'condense_question_prompt',
  'condense_answer_prompt',
  'text_qa_prompt',
  'refine_prompt',
  'intent_graph_knowledge',
  'normal_graph_knowledge',
  'clarifying_question_prompt',
  'generate_goal_prompt',
  'further_questions_prompt',
] as const;

const llmAccessor: { [P in (typeof llmPromptFields[number])]: GeneralSettingsFieldAccessor<ChatEngine, string | null> } = Object.fromEntries(llmPromptFields.map(name => [name, llmOptionAccessor(name)])) as never;
const llmSchema = z.string().nullable();

const postVerificationUrlAccessor = optionAccessor('post_verification_url');
const postVerificationUrlSchema = z.string().nullable();

const postVerificationTokenAccessor = optionAccessor('post_verification_token');
const postVerificationTokenSchema = z.string().nullable();

const externalEngineAccessor: GeneralSettingsFieldAccessor<ChatEngine, string | null> = {
  path: ['engine_options'],
  get (engine) {
    return engine.engine_options.external_engine_config?.stream_chat_api_url ?? null;
  },
  set (engine, value) {
    return {
      ...engine,
      engine_options: {
        ...engine.engine_options,
        external_engine_config: {
          stream_chat_api_url: value,
        },
      },
    };
  },
};
const externalEngineSchema = z.string().nullable();

function Section ({ title, children }: { title: string, children: ReactNode }) {
  return (
    <>
      <SecondaryNavigatorMain className="max-w-screen-sm space-y-8 px-2" value={title} strategy="mount">
        {children}
      </SecondaryNavigatorMain>
    </>
  );
}

function SubSection ({ title, children }: { title: ReactNode, children: ReactNode }) {
  return (
    <section className="space-y-4">
      <h4 className="text-lg">{title}</h4>
      {children}
    </section>
  );
}
