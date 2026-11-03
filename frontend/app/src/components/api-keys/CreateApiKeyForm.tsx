import { createApiKey, type CreateApiKeyResponse } from '@/api/api-keys';
import { FormInput, type FormControlWidgetProps } from '@/components/form/control-widget';
import { withCreateEntityForm } from '@/components/form/create-entity-form';
import { z } from 'zod';
import { KBListSelect } from '@/components/chat-engine/kb-list-select';
import { useAllChatEngines } from '@/components/chat-engine/hooks';
import type { ApiKeyScope } from '@/api/api-keys';

const schema = z.object({
  description: z.string().min(1).max(100),
  knowledge_base_ids: z.array(z.number().int().positive()).max(100),
  chat_engine_ids: z.array(z.number().int().positive()).max(100),
  scopes: z.array(z.enum(['retrieve:chunks', 'retrieve:graph', 'chat:write'])).min(1).max(3),
  expires_in_days: z.coerce.number().int().min(1).max(90),
}).superRefine((value, context) => {
  if (value.scopes.some(scope => scope.startsWith('retrieve:')) && !value.knowledge_base_ids.length) {
    context.addIssue({ code: z.ZodIssueCode.custom, path: ['knowledge_base_ids'], message: 'Select knowledge bases for retrieval.' });
  }
  if (value.scopes.includes('chat:write') && !value.chat_engine_ids.length) {
    context.addIssue({ code: z.ZodIssueCode.custom, path: ['chat_engine_ids'], message: 'Select chat engines for conversations.' });
  }
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
    <FormImpl onCreated={onCreated} defaultValues={{ description: '', knowledge_base_ids: [], chat_engine_ids: [], scopes: ['retrieve:chunks'], expires_in_days: 30 }}>
      <FormImpl.Basic name="description" label="API Key Description">
        <FormInput />
      </FormImpl.Basic>
      <FormImpl.Basic name="knowledge_base_ids" label="Knowledge bases">
        <KBListSelect />
      </FormImpl.Basic>
      <FormImpl.Basic name="chat_engine_ids" label="Chat engines">
        <EngineScopeSelect />
      </FormImpl.Basic>
      <FormImpl.Basic name="scopes" label="Allowed operations">
        <ScopeSelect />
      </FormImpl.Basic>
      <FormImpl.Basic name="expires_in_days" label="Expires in days (1–90)">
        <FormInput type="number" min={1} max={90} />
      </FormImpl.Basic>
      <p className="text-sm text-muted-foreground">API keys can use only the selected knowledge bases and chat engines. Each key can continue only its own conversations. They cannot manage users, settings, or other API keys.</p>
    </FormImpl>
  );
}

function ScopeSelect ({ value = [], onChange, disabled, id }: FormControlWidgetProps<ApiKeyScope[]>) {
  return <fieldset id={id} disabled={disabled} className="space-y-2">
    {([['retrieve:chunks', 'Search document chunks'], ['retrieve:graph', 'Search knowledge graphs'], ['chat:write', 'Create and continue conversations']] as const).map(([scope, label]) => (
      <label key={scope} className="flex items-center gap-2">
        <input type="checkbox" checked={value.includes(scope)} onChange={event => onChange?.(event.target.checked ? [...value, scope] : value.filter(item => item !== scope))} />
        {label}
      </label>
    ))}
  </fieldset>;
}

function EngineScopeSelect ({ value = [], onChange, disabled, id }: FormControlWidgetProps<number[]>) {
  const { data, isLoading, error } = useAllChatEngines();
  if (isLoading) return <span>Loading chat engines...</span>;
  if (error) return <span role="alert">Chat engines could not be loaded.</span>;
  return <fieldset id={id} disabled={disabled} className="space-y-2">
    {data?.map(engine => <label key={engine.id} className="flex items-center gap-2">
      <input type="checkbox" checked={value.includes(engine.id)} onChange={event => onChange?.(event.target.checked ? [...value, engine.id] : value.filter(item => item !== engine.id))} />
      {engine.name}{engine.is_public ? '' : ' (private)'}
    </label>)}
  </fieldset>;
}
