import { uploadFiles } from '@/api/datasources';
import { createEvaluationDataset, type EvaluationDataset } from '@/api/evaluations';
import { FormInput } from '@/components/form/control-widget';
import { withCreateEntityForm } from '@/components/form/create-entity-form';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import { FileInput } from '@/components/form/widgets/FileInput';
import { zodFile } from '@/lib/zod';
import type { ComponentProps } from 'react';
import { z } from 'zod';

const schema = z.object({
  name: z.string().min(1),
  upload_file: zodFile(),
});

const FormImpl = withCreateEntityForm(schema, async ({ upload_file, ...params }) => {
  const [file] = await uploadFiles([upload_file]);
  return await createEvaluationDataset({
    ...params,
    upload_id: file.id,
  });
});

export function CreateEvaluationDatasetForm ({ transitioning, onCreated }: Omit<ComponentProps<typeof FormImpl>, 'defaultValues' | 'children'>) {
  return (
    <FormImpl
      defaultValues={{
        name: '',
      }}
      transitioning={transitioning}
      onCreated={onCreated}
    >
      <FormFieldBasicLayout name="name" label="Name" required>
        <FormInput />
      </FormFieldBasicLayout>
      <FormFieldBasicLayout name="upload_file" label="Upload File" required>
        <FileInput accept={['.csv']} />
      </FormFieldBasicLayout>
    </FormImpl>
  );
}
