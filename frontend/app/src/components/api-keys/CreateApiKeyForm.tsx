import { createApiKey, type CreateApiKeyResponse } from '@/api/api-keys';
import { FormInput, type FormControlWidgetProps } from '@/components/form/control-widget';
import { withCreateEntityForm } from '@/components/form/create-entity-form';
import { z } from 'zod';
import { KBListSelect } from '@/components/chat-engine/kb-list-select';
import type { ApiKeyScope } from '@/api/api-keys';

const schema = z.object({
  description: z.string().min(1).max(100),
  knowledge_base_ids: z.array(z.number().int().positive()).min(1).max(100),
  scopes: z.array(z.enum(['retrieve:chunks', 'retrieve:graph'])).min(1).max(2),
  expires_in_days: z.coerce.number().int().min(1).max(90),
});

export interface CreateApiKeyFormProps {
  onCreated?: (data: CreateApiKeyResponse) => void;
}

const FormImpl = withCreateEntityForm(schema, createApiKey, {
  submitTitle: 'Create API Key',
  submittingTitle: 'Creating API Key...',
});

export function CreateApiKeyForm ({ onCreated }: CreateApiKeyFormProps) {
  return (
    <FormImpl onCreated={onCreated} defaultValues={{ description: '', knowledge_base_ids: [], scopes: ['retrieve:chunks'], expires_in_days: 30 }}>
      <FormImpl.Basic name="description" label="API Key Description">
        <FormInput />
      </FormImpl.Basic>
      <FormImpl.Basic name="knowledge_base_ids" label="Knowledge bases">
        <KBListSelect />
      </FormImpl.Basic>
      <FormImpl.Basic name="scopes" label="Allowed operations">
        <ScopeSelect />
      </FormImpl.Basic>
      <FormImpl.Basic name="expires_in_days" label="Expires in days (1–90)">
        <FormInput type="number" min={1} max={90} />
      </FormImpl.Basic>
      <p className="text-sm text-muted-foreground">API keys can retrieve data from the selected knowledge bases. They cannot manage users, settings, or other API keys.</p>
    </FormImpl>
  );
}

function ScopeSelect ({ value = [], onChange, disabled, id }: FormControlWidgetProps<ApiKeyScope[]>) {
  return <fieldset id={id} disabled={disabled} className="space-y-2">
    {([['retrieve:chunks', 'Search document chunks'], ['retrieve:graph', 'Search knowledge graphs']] as const).map(([scope, label]) => (
      <label key={scope} className="flex items-center gap-2">
        <input type="checkbox" checked={value.includes(scope)} onChange={event => onChange?.(event.target.checked ? [...value, scope] : value.filter(item => item !== scope))} />
        {label}
      </label>
    ))}
  </fieldset>;
}
