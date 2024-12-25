import { createKnowledgeBase } from '@/api/knowledge-base';
import { EmbeddingModelSelect, LLMSelect } from '@/components/form/biz';
import { FormInput, FormTextarea } from '@/components/form/control-widget';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import { FormRootError } from '@/components/form/root-error';
import { handleSubmitHelper } from '@/components/form/utils';
import { FormIndexMethods } from '@/components/knowledge-base/form-index-methods';
import { mutateKnowledgeBases } from '@/components/knowledge-base/hooks';
import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useRouter } from 'next/navigation';
import { useId, useTransition } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

export function CreateKnowledgeBaseForm ({}: {}) {
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();
  const id = useId();

  const form = useForm<z.infer<typeof createKnowledgeBaseParamsSchema>>({
    resolver: zodResolver(createKnowledgeBaseParamsSchema),
    disabled: transitioning,
    defaultValues: {
      name: '',
      description: '',
      index_methods: ['vector'],
      data_sources: [],
    },
  });

  const handleSubmit = handleSubmitHelper(form, async (data) => {
    const kb = await createKnowledgeBase({
      ...data,
      data_sources: [],
    });

    startTransition(() => {
      router.push(`/knowledge-bases/${kb.id}/data-sources`);
      router.refresh();
    });
    void mutateKnowledgeBases();
  });

  return (
    <Form {...form}>
      <form id={id} className="max-w-screen-sm space-y-4" onSubmit={handleSubmit}>
        <FormFieldBasicLayout name="name" label="Name">
          <FormInput placeholder="The name of the knowledge base" />
        </FormFieldBasicLayout>
        <FormFieldBasicLayout name="description" label="Description">
          <FormTextarea placeholder="The description of the knowledge base" />
        </FormFieldBasicLayout>
        <FormFieldBasicLayout name="llm_id" label="LLM" description="Specify the LLM used in building the index. If not specified, the default model will be used.">
          <LLMSelect />
        </FormFieldBasicLayout>
        <FormFieldBasicLayout name="embedding_model_id" label="Embedding Model" description="Specify the embedding model used to convert the corpus into vector embedding. If not specified, the default model will be used.">
          <EmbeddingModelSelect />
        </FormFieldBasicLayout>
        <FormFieldBasicLayout name="index_methods" label="Index Methods">
          <FormIndexMethods />
        </FormFieldBasicLayout>
        <FormRootError />
        <Button type="submit" form={id} disabled={form.formState.disabled || form.formState.isSubmitting || transitioning}>Submit</Button>
      </form>
    </Form>
  );
}

const createKnowledgeBaseParamsSchema = z.object({
  name: z.string().min(1),
  description: z.string(),
  index_methods: z.enum(['knowledge_graph', 'vector']).array(),
  llm_id: z.number().nullable().optional(),
  embedding_model_id: z.number().nullable().optional(),
  data_sources: z.never().array().length(0), // use external form
});
