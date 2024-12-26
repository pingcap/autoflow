import { createEvaluationTask, type CreateEvaluationTaskParams } from '@/api/evaluations';
import { ChatEngineSelect, EvaluationDatasetSelect } from '@/components/form/biz';
import { FormInput } from '@/components/form/control-widget';
import { withCreateEntityForm } from '@/components/form/create-entity-form';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import type { ComponentProps } from 'react';
import { z, type ZodType } from 'zod';

const schema = z.object({
  name: z.string().min(1),
  evaluation_dataset_id: z.number().int(),
  chat_engine: z.string().optional(),
  run_size: z.coerce.number().int().min(1).optional(),
}) satisfies ZodType<CreateEvaluationTaskParams, any, any>;

const FormImpl = withCreateEntityForm(schema, createEvaluationTask);

export function CreateEvaluationTaskForm ({ transitioning, onCreated }: Omit<ComponentProps<typeof FormImpl>, 'defaultValues' | 'children'>) {
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
      <FormFieldBasicLayout name="evaluation_dataset_id" label="Evaluation Dataset" required>
        <EvaluationDatasetSelect />
      </FormFieldBasicLayout>
      <FormFieldBasicLayout name="chat_engine" label="Chat Engine">
        <ChatEngineSelect />
      </FormFieldBasicLayout>
      <FormFieldBasicLayout name="run_size" label="Run Size">
        <FormInput type="number" min={1} step={1} />
      </FormFieldBasicLayout>
    </FormImpl>
  );
}
