import { createApiKey, type CreateApiKeyResponse } from '@/api/api-keys';
import { FormInput } from '@/components/form/control-widget';
import { withCreateEntityFormBeta } from '@/components/form/create-entity-form';
import { FormFieldBasicLayout } from '@/components/form/field-layout.beta';
import { z } from 'zod';

const schema = z.object({
  description: z.string(),
});

export interface CreateApiKeyFormProps {
  onCreated?: (data: CreateApiKeyResponse) => void;
}

const FormImpl = withCreateEntityFormBeta(schema, createApiKey, {
  submitTitle: 'Create API Key',
  submittingTitle: 'Creating API Key...',
});

export function CreateApiKeyForm ({ onCreated }: CreateApiKeyFormProps) {
  return (
    <FormImpl onCreated={onCreated}>
      <FormFieldBasicLayout name="description" label="API Key Description">
        <FormInput />
      </FormFieldBasicLayout>
    </FormImpl>
  );
}
