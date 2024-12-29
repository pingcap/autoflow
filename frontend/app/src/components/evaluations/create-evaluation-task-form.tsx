import { createEvaluationTask, type CreateEvaluationTaskParams } from '@/api/evaluations';
import { ChatEngineSelect, EvaluationDatasetSelect } from '@/components/form/biz';
import { FormInput } from '@/components/form/control-widget';
import { withCreateEntityForm as withCreateEntityForm } from '@/components/form/create-entity-form';
import { formFieldLayout } from '@/components/form/field-layout';
import type { ComponentProps } from 'react';
import { z, type ZodType } from 'zod';

const schema = z.object({
  name: z.string().min(1),
  evaluation_dataset_id: z.number().int(),
  chat_engine: z.string().optional(),
  run_size: z.coerce.number().int().min(1).optional(),
}) satisfies ZodType<CreateEvaluationTaskParams, any, any>;

const FormImpl = withCreateEntityForm(schema, createEvaluationTask);
const field = formFieldLayout<typeof schema>();

export function CreateEvaluationTaskForm ({ transitioning, onCreated }: Omit<ComponentProps<typeof FormImpl>, 'defaultValues' | 'children'>) {
  return (
    <FormImpl
      transitioning={transitioning}
      onCreated={onCreated}
    >
      <field.Basic name="name" label="Name" required defaultValue="">
        <FormInput />
      </field.Basic>
      <field.Basic name="evaluation_dataset_id" label="Evaluation Dataset" required>
        <EvaluationDatasetSelect />
      </field.Basic>
      <field.Basic name="chat_engine" label="Chat Engine">
        <ChatEngineSelect />
      </field.Basic>
      <field.Basic name="run_size" label="Run Size">
        <FormInput type="number" min={1} step={1} />
      </field.Basic>
    </FormImpl>
  );
}
