import { createDatasource, type DeprecatedDatasource } from '@/api/datasources';
import { BasicCreateDatasourceFormLayout } from '@/components/datasource/BasicCreateDatasourceForm';
import { createDatasourceBaseSchema } from '@/components/datasource/schema';
import { FormInput } from '@/components/form/control-widget';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

const schema = createDatasourceBaseSchema.extend({
  url: z.string().url(),
});

export interface CreateWebSitemapDatasourceFormProps {
  transitioning?: boolean;
  onCreated?: (datasource: DeprecatedDatasource) => void;
  excludesLLM?: boolean;
}

export default function CreateWebSitemapDatasourceForm ({ excludesLLM, onCreated, transitioning }: CreateWebSitemapDatasourceFormProps) {
  const form = useForm({
    resolver: zodResolver(schema),
    defaultValues: {
      name: '',
      description: '',
      build_kg_index: false,
      url: '',
      llm_id: null,
    },
  });

  const handleSubmit = form.handleSubmit(async ({ url, ...data }) => {
    const createdDatasource = await createDatasource({
      ...data,
      data_source_type: 'web_sitemap',
      config: { url },
    });
    onCreated?.(createdDatasource);
  });

  return (
    <BasicCreateDatasourceFormLayout form={form} onSubmit={handleSubmit} transitioning={transitioning} excludesLLM={excludesLLM}>
      <FormFieldBasicLayout name="url" label="Sitemap URL">
        <FormInput placeholder="https://example.com/sitemap.xml" />
      </FormFieldBasicLayout>
    </BasicCreateDatasourceFormLayout>
  );
}
