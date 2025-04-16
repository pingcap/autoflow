'use client';

import { type ChatEngineOptions, createChatEngine } from '@/api/chat-engines';
import { KBListSelectForObjectValue } from '@/components/chat-engine/kb-list-select';
import { FormSection, FormSectionsProvider, useFormSectionFields } from '@/components/form-sections';
import { LLMSelect, RerankerSelect } from '@/components/form/biz';
import { FormCheckbox, FormInput, FormSwitch } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { FormRootError } from '@/components/form/root-error';
import { onSubmitHelper } from '@/components/form/utils';
import { PromptInput } from '@/components/form/widgets/PromptInput';
import { SecondaryNavigatorItem, SecondaryNavigatorLayout, SecondaryNavigatorList, SecondaryNavigatorMain } from '@/components/secondary-navigator-list';
import { Button } from '@/components/ui/button';
import { Form, formDomEventHandlers, useFormContext } from '@/components/ui/form.beta';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { useForm } from '@tanstack/react-form';
import { capitalCase } from 'change-case-all';
import { useRouter } from 'next/navigation';
import { type ReactNode, useEffect, useId, useState, useTransition } from 'react';
import { toast } from 'sonner';
import { z } from 'zod';

const schema = z.object({
  name: z.string().min(1),
  llm_id: z.number().optional(),
  fast_llm_id: z.number().optional(),
  reranker_id: z.number().optional(),
  engine_options: z.object({
    knowledge_base: z.object({
      linked_knowledge_bases: z.object({
        id: z.number(),
      }).array().min(1),
    }),
    knowledge_graph: z.object({
      depth: z.number().min(1).nullable().optional(),
    }).passthrough().optional(),
    llm: z.object({}).passthrough().optional(),
  }).passthrough(),
});

const field = formFieldLayout<typeof schema>();

const nameSchema = z.string().min(1);
const kbSchema = z.object({ id: z.number() }).array().min(1);
const kgGraphDepthSchema = z.number().min(1).optional();

export function CreateChatEngineForm ({ defaultChatEngineOptions }: { defaultChatEngineOptions: ChatEngineOptions }) {
  const [transitioning, startTransition] = useTransition();
  const [submissionError, setSubmissionError] = useState<unknown>(undefined);
  const router = useRouter();
  const id = useId();

  const form = useForm({
    onSubmit: onSubmitHelper(schema, async data => {
      const ce = await createChatEngine(data);
      startTransition(() => {
        router.push(`/chat-engines/${ce.id}`);
        router.refresh();
      });
    }, setSubmissionError),
    onSubmitInvalid () {
      toast.error('Validation failed', { description: 'Please check your chat engine configurations.' });
    },
  });

  return (
    <Form form={form} disabled={transitioning} submissionError={submissionError}>
      <FormSectionsProvider>
        <form id={id} {...formDomEventHandlers(form, transitioning)}>
          <SecondaryNavigatorLayout defaultValue="General">
            <SecondaryNavigatorList>
              <SectionTabTrigger required value="General" />
              <SectionTabTrigger required value="Retrieval" />
              <SectionTabTrigger value="Generation" />
              <SectionTabTrigger value="Experimental" />
              <Separator />
              <FormRootError />
              <Button className="w-full" type="submit" form={id} disabled={form.state.isSubmitting || transitioning}>
                Create Chat Engine
              </Button>
            </SecondaryNavigatorList>

            <Section title="General">
              <field.Basic required name="name" label="Name" defaultValue="" validators={{ onSubmit: nameSchema, onBlur: nameSchema }}>
                <FormInput placeholder="Enter chat engine name" />
              </field.Basic>
              <SubSection title="Models">
                <field.Basic name="llm_id" label="LLM">
                  <LLMSelect />
                </field.Basic>
                <field.Basic name="fast_llm_id" label="Fast LLM">
                  <LLMSelect />
                </field.Basic>
              </SubSection>
            </Section>

            <Section title="Retrieval">
              <SubSection title="Knowledge Sources">
                <field.Basic required name="engine_options.knowledge_base.linked_knowledge_bases" label="Knowledge Bases" validators={{ onChange: kbSchema, onSubmit: kbSchema }}>
                  <KBListSelectForObjectValue />
                </field.Basic>
                <field.Inline name="engine_options.hide_sources" label="Hide Sources" description="Hide knowledge sources in chat responses">
                  <FormCheckbox />
                </field.Inline>
              </SubSection>
              <SubSection title="Semantic Search">
                <field.Basic name="reranker_id" label="Reranker">
                  <RerankerSelect />
                </field.Basic>
              </SubSection>
              <SubSection title="Knowledge Graph">
                <field.Contained name="engine_options.knowledge_graph.enabled" label="Enable Knowledge Graph" fallbackValue={defaultChatEngineOptions.knowledge_graph?.enabled} description="Enable knowledge graph functionality to enhance context understanding and relationship mapping between different pieces of information">
                  <FormSwitch />
                </field.Contained>
                <field.Basic name="engine_options.knowledge_graph.depth" label="Depth" fallbackValue={defaultChatEngineOptions.knowledge_graph?.depth} validators={{ onBlur: kgGraphDepthSchema, onSubmit: kgGraphDepthSchema }}>
                  <FormInput type="number" min={1} step={1} />
                </field.Basic>
                <field.Inline name="engine_options.knowledge_graph.include_meta" label="Include Meta" fallbackValue={defaultChatEngineOptions.knowledge_graph?.include_meta} description="Include metadata information in knowledge graph nodes to provide additional context">
                  <FormCheckbox />
                </field.Inline>
                <field.Inline name="engine_options.knowledge_graph.with_degree" label="With Degree" fallbackValue={defaultChatEngineOptions.knowledge_graph?.with_degree} description="Include connection degree information in knowledge graph relationships">
                  <FormCheckbox />
                </field.Inline>
                <field.Inline name="engine_options.knowledge_graph.using_intent_search" label="Using intent search" fallbackValue={defaultChatEngineOptions.knowledge_graph?.using_intent_search} description="Enable intent-based search in knowledge graph traversal for more contextually relevant results">
                  <FormCheckbox />
                </field.Inline>
                {(['intent_graph_knowledge', 'normal_graph_knowledge'] as const).map(name => (
                  <field.Basic key={name} name={`engine_options.llm.${name}`} label={capitalCase(name)} fallbackValue={defaultChatEngineOptions.llm?.[name]} description={llmPromptDescriptions[name]}>
                    <PromptInput />
                  </field.Basic>
                ))}
              </SubSection>
            </Section>

            <Section title="Generation">
              <SubSection title="Clarify Question">
                <field.Contained unimportant name="engine_options.clarify_question" label="Clarify Question" description="Enable the system to ask clarifying questions when user input is ambiguous">
                  <FormSwitch />
                </field.Contained>
                <field.Basic name="engine_options.llm.clarifying_question_prompt" label="Clarify Question Prompt" fallbackValue={defaultChatEngineOptions.llm?.clarifying_question_prompt} description={llmPromptDescriptions.clarifying_question_prompt}>
                  <PromptInput />
                </field.Basic>
              </SubSection>
              <SubSection title="Rewrite Question">
                <field.Basic name="engine_options.llm.condense_question_prompt" label="Rewrite Question Prompt" fallbackValue={defaultChatEngineOptions.llm?.condense_question_prompt} description={llmPromptDescriptions.condense_question_prompt}>
                  <PromptInput />
                </field.Basic>
              </SubSection>
              <SubSection title="Answer Question">
                <field.Basic name="engine_options.llm.text_qa_prompt" label="Answer Question Prompt" fallbackValue={defaultChatEngineOptions.llm?.text_qa_prompt} description={llmPromptDescriptions.text_qa_prompt}>
                  <PromptInput />
                </field.Basic>
              </SubSection>
              <SubSection title="Recommend More Questions">
                <field.Basic name="engine_options.llm.further_questions_prompt" label="Further Questions Prompt" fallbackValue={defaultChatEngineOptions.llm?.further_questions_prompt} description={llmPromptDescriptions.further_questions_prompt}>
                  <PromptInput />
                </field.Basic>
              </SubSection>
            </Section>

            <Section title="Experimental">
              <SubSection title="External Engine">
                <field.Basic name="engine_options.external_engine_config.stream_chat_api_url" label="External Chat Engine API URL (StackVM)" fallbackValue={defaultChatEngineOptions.external_engine_config?.stream_chat_api_url ?? ''}>
                  <FormInput />
                </field.Basic>
                <field.Basic name="engine_options.llm.generate_goal_prompt" label="Generate Goal Prompt" fallbackValue={defaultChatEngineOptions.llm?.generate_goal_prompt} description={llmPromptDescriptions.generate_goal_prompt}>
                  <PromptInput />
                </field.Basic>
              </SubSection>
              <SubSection title="Post Verification">
                <field.Basic name="engine_options.post_verification_url" label="Post Verifycation Service URL" fallbackValue={defaultChatEngineOptions.post_verification_url ?? ''}>
                  <FormInput />
                </field.Basic>
                <field.Basic name="engine_options.post_verification_token" label="Post Verifycation Service Token" fallbackValue={defaultChatEngineOptions.post_verification_token ?? ''}>
                  <FormInput />
                </field.Basic>
              </SubSection>
            </Section>
          </SecondaryNavigatorLayout>
        </form>
      </FormSectionsProvider>
    </Form>
  );
}

function SectionTabTrigger ({ value, required }: { value: string, required?: boolean }) {
  const [invalid, setInvalid] = useState(false);
  const { form } = useFormContext();
  const fields = useFormSectionFields(value);

  useEffect(() => {
    return form.store.subscribe(() => {
      let invalid = false;
      for (let field of fields.values()) {
        if (field.getMeta().errors.length > 0) {
          invalid = true;
          break;
        }
      }
      setInvalid(invalid);
    });
  }, [form, fields, value]);

  return (
    <SecondaryNavigatorItem value={value}>
      <span className={cn(invalid && 'text-destructive')}>
        {value}
      </span>
      {required && <sup className="text-destructive" aria-hidden>*</sup>}
    </SecondaryNavigatorItem>
  );
}

function Section ({ title, children }: { title: string, children: ReactNode }) {
  return (
    <FormSection value={title}>
      <SecondaryNavigatorMain className="space-y-8 max-w-screen-sm px-2" value={title} strategy="hidden">
        {children}
      </SecondaryNavigatorMain>
    </FormSection>
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

const llmPromptFields = [
  'condense_question_prompt',
  'text_qa_prompt',
  'intent_graph_knowledge',
  'normal_graph_knowledge',
  'clarifying_question_prompt',
  'generate_goal_prompt',
  'further_questions_prompt',
] as const;

const llmPromptDescriptions: { [P in typeof llmPromptFields[number]]: string } = {
  'condense_question_prompt': 'Template for condensing a conversation history and follow-up question into a standalone question',
  'text_qa_prompt': 'Template for generating answers based on provided context and question',
  'intent_graph_knowledge': 'Template for extracting knowledge from intent-based graph traversal',
  'normal_graph_knowledge': 'Template for processing knowledge from standard graph traversal',
  'clarifying_question_prompt': 'Template for generating clarifying questions when user input is ambiguous',
  'generate_goal_prompt': 'Template for generating conversation goals based on user input',
  'further_questions_prompt': 'Template for generating follow-up questions to continue the conversation',
}; 