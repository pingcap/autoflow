import type { ComponentProps } from 'react';
import { FileInput } from '@/components/form/widgets/FileInput';
import { FormInput } from '@/components/form/control-widget';
import Link from 'next/link';
import { createEvaluationDataset } from '@/api/evaluations';
import { formFieldLayout } from '@/components/form/field-layout';
import { uploadFiles } from '@/api/datasources';
import { withCreateEntityForm } from '@/components/form/create-entity-form';
import { z } from 'zod';
import { zodFile } from '@/lib/zod';

const schema = z.object({
  name: z.string().min(1),
  upload_file: zodFile().optional(),
});

const field = formFieldLayout<typeof schema>();

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
      <field.Basic name="name" label="Name" required>
        <FormInput />
      </field.Basic>
      <field.Basic name="upload_file" label="Upload File" description={<>Evaluation dataset CSV file. See the <Link className='underline' href='https://autoflow.tidb.ai/evaluation#prerequisites'>documentation</Link> for the format.</>}>
        <FileInput accept={['.csv']} />
      </field.Basic>
    </FormImpl>
  );
}
