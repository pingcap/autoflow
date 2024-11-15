import { createDatasource, type DeprecatedDatasource } from '@/api/datasources';
import { BasicCreateDatasourceFormLayout } from '@/components/datasource/BasicCreateDatasourceForm';
import { createDatasourceBaseSchema } from '@/components/datasource/schema';
import { FormInput } from '@/components/form/control-widget';
import { FormPrimitiveArrayFieldBasicLayout } from '@/components/form/field-layout';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

const schema = createDatasourceBaseSchema.extend({
  urls: z.string().url().array().min(1),
});

export interface CreateWebSinglePageDatasourceFormProps {
  excludesLLM?: boolean;
  transitioning?: boolean;
  onCreated?: (datasource: DeprecatedDatasource) => void;
}

export default function CreateWebSinglePageDatasourceForm ({ excludesLLM, transitioning, onCreated }: CreateWebSinglePageDatasourceFormProps) {
  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
    defaultValues: {
      name: '',
      description: '',
      build_kg_index: false,
      urls: [''],
      llm_id: null,
    },
  });

  const handleSubmit = form.handleSubmit(async ({ urls, ...data }) => {
    const createdDatasource = await createDatasource({
      ...data,
      data_source_type: 'web_single_page',
      config: { urls },
    });
    onCreated?.(createdDatasource);
  });

  return (
    <BasicCreateDatasourceFormLayout form={form} onSubmit={handleSubmit} transitioning={transitioning} excludesLLM={excludesLLM}>
      <FormPrimitiveArrayFieldBasicLayout name="urls" label="Page URL" defaultValue={() => ''}>
        <FormInput placeholder="https://example.com/" />
      </FormPrimitiveArrayFieldBasicLayout>
    </BasicCreateDatasourceFormLayout>
  );
}
