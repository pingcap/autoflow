import { uploadFiles } from '@/api/datasources';
import { createEvaluationDataset } from '@/api/evaluations';
import { FormInput } from '@/components/form/control-widget';
import { withCreateEntityForm } from '@/components/form/create-entity-form';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import { FileInput } from '@/components/form/widgets/FileInput';
import { zodFile } from '@/lib/zod';
import type { ComponentProps } from 'react';
import { z } from 'zod';

const schema = z.object({
  name: z.string().min(1),
  upload_file: zodFile().optional(),
});

const FormImpl = withCreateEntityForm(schema, async ({ upload_file, ...params }) => {
  if (upload_file != null) {
    const [file] = await uploadFiles([upload_file]);
    return await createEvaluationDataset({
      ...params,
      upload_id: file.id,
    });
  } else {
    return await createEvaluationDataset({
      ...params,
    });
  }
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
      <FormFieldBasicLayout name="upload_file" label="Upload File">
        <FileInput accept={['.csv']} />
      </FormFieldBasicLayout>
    </FormImpl>
  );
}
