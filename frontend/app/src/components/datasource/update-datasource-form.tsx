import { type Datasource, updateDatasource } from '@/api/datasources';
import { FormInput } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { onSubmitHelper } from '@/components/form/utils';
import { Button } from '@/components/ui/button';
import { Form, formDomEventHandlers } from '@/components/ui/form.beta';
import { useForm } from '@tanstack/react-form';
import { useState } from 'react';
import { z } from 'zod';

const schema = z.object({
  name: z.string().min(1, 'Must not empty'),
});

const field = formFieldLayout<typeof schema>();

export function UpdateDatasourceForm ({ knowledgeBaseId, datasource, onUpdated }: { knowledgeBaseId: number, datasource: Datasource, onUpdated?: () => void }) {
  const [submissionError, setSubmissionError] = useState<unknown>(undefined);

  const form = useForm<UpdateDatasourceFormParams>({
    validators: {
      onSubmit: schema,
    },
    defaultValues: {
      name: datasource.name,
    },
    onSubmit: onSubmitHelper(schema, async data => {
      await updateDatasource(knowledgeBaseId, datasource.id, data);
      onUpdated?.();
    }, setSubmissionError),
  });

  return (
    <Form form={form} submissionError={submissionError}>
      <form className="space-y-4" {...formDomEventHandlers(form)}>
        <field.Basic name="name" label="Name">
          <FormInput />
        </field.Basic>
        <Button type="submit" disabled={form.state.isSubmitting}>
          Update
        </Button>
      </form>
    </Form>
  );
}

interface UpdateDatasourceFormParams {
  name: string;
}
