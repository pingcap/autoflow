'use client';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { getErrorMessage } from '@/lib/errors';
import { modelConnectionUpdate, type ModelConnectionUpdate } from '@/lib/model-connection-update';
import { useEffect, useId, useState } from 'react';

export function ModelConnectionForm ({ config, credentialType, disabled = false, onSave }: {
  config: unknown;
  credentialType?: string;
  disabled?: boolean;
  onSave: (value: ModelConnectionUpdate) => Promise<unknown>;
}) {
  const id = useId();
  const original = JSON.stringify(config ?? {}, undefined, 2);
  const [configText, setConfigText] = useState(original);
  const [credential, setCredential] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { setConfigText(original); }, [original]);
  const locked = disabled || saving || !credentialType;

  return (
    <form className="space-y-4" onSubmit={async event => {
      event.preventDefault();
      setError('');
      setSaving(true);
      try {
        await onSave(modelConnectionUpdate(configText, credential, credentialType ?? 'str'));
        setCredential('');
      } catch (error) {
        setError(getErrorMessage(error));
      } finally {
        setSaving(false);
      }
    }}>
      <div className="space-y-2">
        <label htmlFor={`${id}-config`} className="text-sm font-medium">Model configuration</label>
        <Textarea id={`${id}-config`} value={configText} onChange={event => setConfigText(event.target.value)} rows={8} maxLength={65536} disabled={locked} className="font-mono" />
      </div>
      <div className="space-y-2">
        <label htmlFor={`${id}-credentials`} className="text-sm font-medium">New credentials</label>
        <p className="text-sm text-muted-foreground">Enter new credentials when changing the endpoint. Leave blank to keep existing credentials for other changes.</p>
        {credentialType === 'dict'
          ? <Textarea id={`${id}-credentials`} value={credential} onChange={event => setCredential(event.target.value)} rows={4} maxLength={65536} disabled={locked} autoComplete="off" spellCheck={false} placeholder="Credentials JSON" />
          : <Input id={`${id}-credentials`} type="password" value={credential} onChange={event => setCredential(event.target.value)} maxLength={65536} disabled={locked} autoComplete="new-password" />}
      </div>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      <Button type="submit" disabled={locked || (configText === original && !credential)}>{saving ? 'Saving' : 'Save model connection'}</Button>
    </form>
  );
}
