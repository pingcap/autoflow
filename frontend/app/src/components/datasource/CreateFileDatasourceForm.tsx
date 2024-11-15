import { createDatasource, type DeprecatedDatasource, uploadFiles } from '@/api/datasources';
import { BasicCreateDatasourceFormLayout } from '@/components/datasource/BasicCreateDatasourceForm';
import { createDatasourceBaseSchema } from '@/components/datasource/schema';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import { FilesInput } from '@/components/form/widgets/FilesInput';
import { zodFile } from '@/lib/zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';

const schema = createDatasourceBaseSchema.extend({
  files: zodFile().array().min(1),
});

export interface CreateFileDatasourceFormProps {
  excludesLLM?: boolean;
  transitioning?: boolean;
  onCreated?: (datasource: DeprecatedDatasource) => void;
}

export default function CreateFileDatasourceForm ({ excludesLLM, transitioning, onCreated }: CreateFileDatasourceFormProps) {
  const form = useForm({
    resolver: zodResolver(schema),
    defaultValues: {
      name: '',
      description: '',
      files: [],
      build_kg_index: false,
      llm_id: null,
    },
  });

  const handleSubmit = form.handleSubmit(async ({ files, ...data }) => {
    const uploadedFiles = await uploadFiles(files);
    const createdDatasource = await createDatasource({
      ...data,
      data_source_type: 'file',
      config: uploadedFiles.map(file => ({
        file_name: file.name,
        file_id: file.id,
      })),
    });
    onCreated?.(createdDatasource);
  });

  return (
    <BasicCreateDatasourceFormLayout form={form} onSubmit={handleSubmit} transitioning={transitioning} excludesLLM={excludesLLM}>
      <FormFieldBasicLayout name="files" label="Files" description="Currently support Markdown (*.md), PDF (*.pdf), Microsoft Word (*.docx), Microsoft PowerPoint (*.pptx), Microsoft Excel (*.xlsx) and Text (*.txt) files.">
        <FilesInput accept={['text/plain', 'application/pdf', '.md', '.docx', '.pptx', '.xlsx']} />
      </FormFieldBasicLayout>
    </BasicCreateDatasourceFormLayout>
  );
}
