'use client';

import { type ChatEngineOptions, createChatEngine } from '@/api/chat-engines';
import { FormSection, FormSectionsProvider, useFormSectionFields } from '@/components/form-sections';
import { KBSelect, LLMSelect, RerankerSelect } from '@/components/form/biz';
import { FormCheckbox, FormInput, FormSwitch } from '@/components/form/control-widget';
import { FormFieldBasicLayout, FormFieldContainedLayout, FormFieldInlineLayout } from '@/components/form/field-layout';
import { FormRootError } from '@/components/form/root-error';
import { handleSubmitHelper } from '@/components/form/utils';
import { PromptInput } from '@/components/form/widgets/PromptInput';
import { SecondaryNavigatorItem, SecondaryNavigatorLayout, SecondaryNavigatorList, SecondaryNavigatorMain } from '@/components/secondary-navigator-list';
import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import { capitalCase } from 'change-case-all';
import { useRouter } from 'next/navigation';
import { type ReactNode, useId, useMemo, useTransition } from 'react';
import { get, useForm, useFormContext } from 'react-hook-form';
import { toast } from 'sonner';
import { z } from 'zod';

const schema = z.object({
  name: z.string().min(1),
  llm_id: z.number().optional(),
  fast_llm_id: z.number().optional(),
  reranker_id: z.number().optional(),
  engine_options: z.object({
    knowledge_base: z.object({
      linked_knowledge_base: z.object({
        id: z.number(),
      }),
    }),
    knowledge_graph: z.object({
      depth: z.string().pipe(z.coerce.number().min(1)).optional(),
    }).passthrough(),
    llm: z.object({}).passthrough(),
  }).passthrough(),
});

export function CreateChatEngineForm ({ defaultChatEngineOptions }: { defaultChatEngineOptions: ChatEngineOptions }) {
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();
  const id = useId();

  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
    disabled: transitioning,
    defaultValues: {
      name: '',
      engine_options: {},
    },
  });

  const handleSubmit = handleSubmitHelper(form, async data => {
    // TODO: refactor types
    const ce = await createChatEngine(data as never);
    startTransition(() => {
      router.push(`/chat-engines/${ce.id}`);
      router.refresh();
    });
  }, () => {
    toast.error('Validation failed', { description: 'Please check your chat engine configurations.' });
  });

  return (
    <Form {...form}>
      <FormSectionsProvider>
        <form id={id} onSubmit={handleSubmit}>
          <SecondaryNavigatorLayout defaultValue="Info">
            <SecondaryNavigatorList>
              <SectionTabTrigger required value="Info" />
              <SectionTabTrigger required value="Retrieval" />
              <SectionTabTrigger value="Generation" />
              <SectionTabTrigger value="Features" />
              <Separator />
              <FormRootError />
              <Button className="w-full" type="submit" form={id} disabled={form.formState.isSubmitting || form.formState.disabled}>
                Create Chat Engine
              </Button>
            </SecondaryNavigatorList>

            <Section title="Info">
              <FormFieldBasicLayout required name="name" label="Name">
                <FormInput />
              </FormFieldBasicLayout>
              <SubSection title="Models">
                <FormFieldBasicLayout name="llm_id" label="LLM">
                  <LLMSelect />
                </FormFieldBasicLayout>
                <FormFieldBasicLayout name="fast_llm_id" label="Fast LLM">
                  <LLMSelect />
                </FormFieldBasicLayout>
              </SubSection>
              <SubSection title="External Engine Config">
                <FormFieldBasicLayout name="engine_options.external_engine_config.stream_chat_api_url" label="External Chat Engine API URL (StackVM)" fallbackValue={defaultChatEngineOptions.external_engine_config?.stream_chat_api_url ?? ''}>
                  <FormInput />
                </FormFieldBasicLayout>
                <FormFieldBasicLayout name="engine_options.llm.generate_goal_prompt" label="Generate Goal Prompt" fallbackValue={defaultChatEngineOptions.llm?.generate_goal_prompt} description={llmPromptDescriptions.generate_goal_prompt}>
                  <PromptInput />
                </FormFieldBasicLayout>
              </SubSection>
            </Section>
            <Section title="Retrieval">
              <FormFieldBasicLayout required name="engine_options.knowledge_base.linked_knowledge_base.id" label="Select Knowledge Base">
                <KBSelect />
              </FormFieldBasicLayout>
              <FormFieldBasicLayout name="reranker_id" label="Reranker">
                <RerankerSelect />
              </FormFieldBasicLayout>
              <SubSection title="Knowledge Graph">
                <FormFieldContainedLayout name="engine_options.knowledge_graph.enabled" label="Enable Knowledge Graph" fallbackValue={defaultChatEngineOptions.knowledge_graph?.enabled} description="/// Description TBD">
                  <FormSwitch />
                </FormFieldContainedLayout>
                <FormFieldBasicLayout name="engine_options.knowledge_graph.depth" label="Depth" fallbackValue={defaultChatEngineOptions.knowledge_graph?.depth}>
                  <FormInput type="number" min={1} step={1} />
                </FormFieldBasicLayout>
                <FormFieldInlineLayout name="engine_options.knowledge_graph.include_meta" label="Include Meta" fallbackValue={defaultChatEngineOptions.knowledge_graph?.include_meta} description="/// Description TBD">
                  <FormCheckbox />
                </FormFieldInlineLayout>
                <FormFieldInlineLayout name="engine_options.knowledge_graph.with_degree" label="With Degree" fallbackValue={defaultChatEngineOptions.knowledge_graph?.with_degree} description="/// Description TBD">
                  <FormCheckbox />
                </FormFieldInlineLayout>
                <FormFieldInlineLayout name="engine_options.knowledge_graph.using_intent_search" label="Using intent search" fallbackValue={defaultChatEngineOptions.knowledge_graph?.using_intent_search} description="/// Description TBD">
                  <FormCheckbox />
                </FormFieldInlineLayout>
                {(['intent_graph_knowledge', 'normal_graph_knowledge'] as const).map(field => (
                  <FormFieldBasicLayout key={field} name={`engine_options.llm.${field}`} label={capitalCase(field)} fallbackValue={defaultChatEngineOptions.llm?.[field]} description={llmPromptDescriptions[field]}>
                    <PromptInput />
                  </FormFieldBasicLayout>
                ))}
              </SubSection>
            </Section>
            <Section title="Generation">
              {(['condense_question_prompt', 'condense_answer_prompt', 'text_qa_prompt', 'refine_prompt'] as const).map(field => (
                <FormFieldBasicLayout key={field} name={`engine_options.llm.${field}`} label={capitalCase(field)} fallbackValue={defaultChatEngineOptions.llm?.[field]} description={llmPromptDescriptions[field]}>
                  <PromptInput />
                </FormFieldBasicLayout>
              ))}
            </Section>
            <Section title="Features">
              <FormFieldInlineLayout name="engine_options.hide_sources" label="Hide Reference Sources" description="/// Description TBD">
                <FormCheckbox />
              </FormFieldInlineLayout>
              <SubSection title="Clarify Question">
                <FormFieldContainedLayout unimportant name="engine_options.clarify_question" label="Clarify Question" description="/// Description TBD">
                  <FormSwitch />
                </FormFieldContainedLayout>
                <FormFieldBasicLayout name="engine_options.llm.clarifying_question_prompt" label="Clarifying Question Prompt" fallbackValue={defaultChatEngineOptions.llm?.clarifying_question_prompt} description={llmPromptDescriptions.clarifying_question_prompt}>
                  <PromptInput />
                </FormFieldBasicLayout>
              </SubSection>
              <SubSection title="Post Verification">
                <FormFieldBasicLayout name="engine_options.post_verification_url" label="Post Verifycation Service URL" fallbackValue={defaultChatEngineOptions.post_verification_url ?? ''}>
                  <FormInput />
                </FormFieldBasicLayout>
                <FormFieldBasicLayout name="engine_options.post_verification_token" label="Post Verifycation Service Token" fallbackValue={defaultChatEngineOptions.post_verification_token ?? ''}>
                  <FormInput />
                </FormFieldBasicLayout>
              </SubSection>
              <SubSection title="Further Recommended Questions">
                <FormFieldBasicLayout name="engine_options.llm.further_questions_prompt" label="Further Questions Prompt" fallbackValue={defaultChatEngineOptions.llm?.further_questions_prompt} description={llmPromptDescriptions.further_questions_prompt}>
                  <PromptInput />
                </FormFieldBasicLayout>
              </SubSection>
            </Section>
          </SecondaryNavigatorLayout>
        </form>
      </FormSectionsProvider>
    </Form>
  );
}

function SectionTabTrigger ({ value, required }: { value: string, required?: boolean }) {
  const fields = useFormSectionFields(value);
  const form = useFormContext();

  const validated = useMemo(() => {
    let validated = true;

    for (let field of Array.from(fields)) {
      const error = get(form.formState.errors, field);
      if (error) {
        validated = false;
        break;
      }
    }
    return validated;
  }, [form.formState.errors, fields]);

  return (
    <SecondaryNavigatorItem value={value}>
      <span className={cn(!validated && 'text-destructive')}>
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
  'condense_answer_prompt',
  'text_qa_prompt',
  'refine_prompt',
  'intent_graph_knowledge',
  'normal_graph_knowledge',
  'clarifying_question_prompt',
  'generate_goal_prompt',
  'further_questions_prompt',
] as const;

const llmPromptDescriptions: { [P in typeof llmPromptFields[number]]: string } = {
  'condense_question_prompt': '/// Description TBD',
  'condense_answer_prompt': '/// Description TBD',
  'text_qa_prompt': '/// Description TBD',
  'refine_prompt': '/// Description TBD',
  'intent_graph_knowledge': '/// Description TBD',
  'normal_graph_knowledge': '/// Description TBD',
  'clarifying_question_prompt': '/// Description TBD',
  'generate_goal_prompt': '/// Description TBD',
  'further_questions_prompt': '/// Description TBD',
};
