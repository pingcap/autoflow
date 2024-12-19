import { createEvaluationDatasetItem, type EvaluationDatasetItem } from '@/api/evaluations';
import { FormTextarea } from '@/components/form/control-widget';
import { withCreateEntityFormBeta } from '@/components/form/create-entity-form';
import { FormFieldBasicLayout, FormPrimitiveArrayFieldBasicLayout } from '@/components/form/field-layout.beta';
import { CodeInput } from '@/components/form/widgets/CodeInput';
import { zodJson } from '@/lib/zod';
import { useMemo } from 'react';
import { z } from 'zod';

const schema = z.object({
  query: z.string().min(1),
  reference: z.string().min(1),
  retrieved_contexts: z.string().min(1).array(),
  extra: z.string()
    .pipe(z.custom<string>(s => {
      if (!s.trim()) return true;
      try {
        JSON.parse(s);
        return true;
      } catch {
        return false;
      }
    }, 'Invalid JSON'))
    .transform(s => {
      if (s.trim()) {
        return JSON.parse(s);
      } else {
        return undefined;
      }
    })
    .pipe(zodJson()),
});

export function CreateEvaluationDatasetItemForm ({ evaluationDatasetId, transitioning, onCreated }: { evaluationDatasetId: number, transitioning?: boolean, onCreated?: (item: EvaluationDatasetItem) => void }) {
  const FormImpl = useMemo(() => withCreateEntityFormBeta(schema, params => createEvaluationDatasetItem(evaluationDatasetId, params)), [evaluationDatasetId]);

  return (
    <FormImpl
      defaultValues={{
        query: '',
        reference: '',
        retrieved_contexts: [],
        extra: '{}',
      }}
      transitioning={transitioning}
      onCreated={onCreated}
    >
      <FormFieldBasicLayout name="query" label="Query" required>
        <FormTextarea />
      </FormFieldBasicLayout>
      <FormFieldBasicLayout name="reference" label="Reference" required>
        <FormTextarea />
      </FormFieldBasicLayout>
      <FormPrimitiveArrayFieldBasicLayout name="retrieved_contexts" label="Retrieved Contexts" defaultValue={() => ''} required>
        <FormTextarea />
      </FormPrimitiveArrayFieldBasicLayout>
      <FormFieldBasicLayout name="extra" label="Extra">
        <CodeInput language="json" />
      </FormFieldBasicLayout>
    </FormImpl>
  );
}
