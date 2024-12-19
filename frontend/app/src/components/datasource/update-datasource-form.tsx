import { type Datasource, updateDatasource } from '@/api/datasources';
import { FormInput } from '@/components/form/control-widget';
import { FormFieldBasicLayout } from '@/components/form/field-layout';
import { handleSubmitHelper } from '@/components/form/utils';
import { Button } from '@/components/ui/button';
import { Form } from '@/components/ui/form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

export function UpdateDatasourceForm ({ knowledgeBaseId, datasource, onUpdated }: { knowledgeBaseId: number, datasource: Datasource, onUpdated?: () => void }) {
  const form = useForm<UpdateDatasourceFormParams>({
    resolver: zodResolver(schema),
    defaultValues: {
      name: datasource.name,
    },
  });

  const handleSubmit = handleSubmitHelper(form, async data => {
    await updateDatasource(knowledgeBaseId, datasource.id, data);
    onUpdated?.();
  });

  return (
    <Form {...form}>
      <form className="space-y-4" onSubmit={handleSubmit}>
        <FormFieldBasicLayout name="name" label="Name">
          <FormInput />
        </FormFieldBasicLayout>
        <Button type="submit" disabled={form.formState.disabled || form.formState.isSubmitting}>
          Update
        </Button>
      </form>
    </Form>
  );
}

interface UpdateDatasourceFormParams {
  name: string;
}

const schema = z.object({
  name: z.string().min(1, 'Must not empty'),
});