import { uploadFiles } from '@/api/datasources';
import { createKnowledgeBase } from '@/api/knowledge-base';
import { EmbeddingModelSelect, LLMSelect } from '@/components/form/biz';
import { FormInput, FormTextarea } from '@/components/form/control-widget';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import { handleSubmitHelper } from '@/components/form/utils';
import { createDatasourceSchema, FormCreateDataSources } from '@/components/knowledge-base/form-create-data-sources';
import { FormIndexMethods } from '@/components/knowledge-base/form-index-methods';
import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

export interface CreateKnowledgeBaseFormProps {
}

export function CreateKnowledgeBaseForm ({}: {}) {
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();

  const form = useForm<z.infer<typeof createKnowledgeBaseParamsSchema>>({
    resolver: zodResolver(createKnowledgeBaseParamsSchema),
    defaultValues: {
      name: '',
      description: '',
      index_methods: ['vector'],
      data_sources: [
        { name: '', description: '', data_source_type: 'file', files: [] },
      ],
    },
  });

  const handleSubmit = handleSubmitHelper(form, async (data) => {
    const dataSources = await Promise.all(data.data_sources.map(async (ds) => {
      switch (ds.data_source_type) {
        case 'file': {
          const { files, ...rest } = ds;
          const uploadedFiles = await uploadFiles(ds.files);
          return {
            ...rest,
            config: uploadedFiles.map(f => ({
              file_id: f.id,
              file_name: f.name,
            })),
          };
        }
        case 'web_single_page': {
          const { urls, ...rest } = ds;

          return {
            ...rest,
            config: { urls },
          };
        }

        case 'web_sitemap':
          const { url, ...rest } = ds;

          return {
            ...rest,
            config: { url },
          };
      }
    }));

    const kb = await createKnowledgeBase({
      ...data,
      data_sources: dataSources,
    });

    startTransition(() => {
      router.push(`/knowledge-bases/${kb.id}`);
    });
  });

  return (
    <Form {...form}>
      <form className="space-y-4" onSubmit={handleSubmit}>
        <FormFieldBasicLayout name="name" label="Name">
          <FormInput placeholder="The name of the knowledge base"/>
        </FormFieldBasicLayout>
        <FormFieldBasicLayout name="description" label="Description">
          <FormTextarea placeholder="The description of the knowledge base"/>
        </FormFieldBasicLayout>
        <FormFieldBasicLayout name="llm_id" label="LLM" description="Specify the LLM used in building the index. If not specified, the default model will be used.">
          <LLMSelect />
        </FormFieldBasicLayout>
        <FormFieldBasicLayout name="embedding_model_id" label="Embedding Model" description="Specify the embedding model used to convert the corpus into vector embedding. If not specified, the default model will be used.">
          <EmbeddingModelSelect />
        </FormFieldBasicLayout>
        <FormCreateDataSources />
        <FormFieldBasicLayout name="index_methods" label="Index Methods">
          <FormIndexMethods />
        </FormFieldBasicLayout>
        <Button type="submit">Submit</Button>
      </form>
    </Form>
  );
}

const createKnowledgeBaseParamsSchema = z.object({
  name: z.string(),
  description: z.string(),
  index_methods: z.enum(['knowledge_graph', 'vector']).array(),
  llm_id: z.number().nullable().optional(),
  embedding_model_id: z.number().nullable().optional(),
  data_sources: createDatasourceSchema.array(), // use external form
});
