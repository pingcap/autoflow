'use client';

import type { AllSettings } from '@/api/site-settings';
import { SettingsField } from '@/components/settings/SettingsField';
import { getLangfuseConfiguration, updateLangfuseConfiguration, type LangfuseConfiguration } from '@/api/langfuse';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { getErrorMessage } from '@/lib/errors';
import { useState, type FormEvent } from 'react';
import useSWR from 'swr';

export function IntegrationsSettings ({ schema, showPostVerificationSettings }: { schema: AllSettings, showPostVerificationSettings: boolean }) {
  return (
    <div className="space-y-8 max-w-screen-md">
      <LangfuseSettings schema={schema} />
      {showPostVerificationSettings && <ExperimentalPostVerificationSettings schema={schema} />}
    </div>
  );
}

export function LangfuseSettings ({ hideTitle, disabled, onChanged }: { schema: AllSettings, hideTitle?: boolean, disabled?: boolean, onChanged?: () => void }) {
  const { data, error, mutate } = useSWR('admin-langfuse-configuration', getLangfuseConfiguration);
  return (
    <section className="space-y-6">
      {!hideTitle && <h2 className="text-lg font-medium">Langfuse</h2>}
      {error && <p role="alert">Unable to load Langfuse settings.</p>}
      {!data && !error && <p>Loading Langfuse settings…</p>}
      {data && <LangfuseConfigurationForm key={data.revision} configuration={data} disabled={disabled} onReload={() => mutate()} onSaved={async updated => {
        await mutate(updated, false);
        onChanged?.();
      }} />}
    </section>
  );
}

export function LangfuseConfigurationForm ({ configuration, disabled, onSaved, onReload }: {
  configuration: LangfuseConfiguration;
  disabled?: boolean;
  onSaved: (configuration: LangfuseConfiguration) => Promise<void>;
  onReload: () => Promise<unknown>;
}) {
  const [host, setHost] = useState(configuration.host);
  const [publicKey, setPublicKey] = useState(configuration.public_key);
  const [secret, setSecret] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const destinationChanged = host !== configuration.host || publicKey !== configuration.public_key;

  async function save (event?: FormEvent, clearSecret = false) {
    event?.preventDefault();
    setSaving(true);
    setError('');
    try {
      const updated = await updateLangfuseConfiguration({
        host: clearSecret ? configuration.host : host,
        public_key: clearSecret ? configuration.public_key : publicKey,
        revision: configuration.revision,
        ...(clearSecret ? { clear_secret: true } : secret ? { secret_key: secret } : {}),
      });
      setSecret('');
      await onSaved(updated);
    } catch (error) {
      setError(getErrorMessage(error));
    } finally {
      setSaving(false);
    }
  }

  return <form className="space-y-4" onSubmit={save}>
    <p className="text-sm text-muted-foreground">{configuration.configured ? 'A secret key is configured. Saved secrets are never shown.' : 'No secret key is configured.'}</p>
    {configuration.migration_required && <p role="alert">The existing key needs encryption. Run the migration before starting workers, or replace the key here.</p>}
    <label className="block space-y-2">Host
      <Input value={host} onChange={event => setHost(event.target.value)} disabled={disabled || saving} maxLength={2048} required />
    </label>
    <label className="block space-y-2">Public key
      <Input value={publicKey} onChange={event => setPublicKey(event.target.value)} disabled={disabled || saving} maxLength={512} />
    </label>
    <label className="block space-y-2">Secret key
      <Input type="password" autoComplete="new-password" value={secret} onChange={event => setSecret(event.target.value)} disabled={disabled || saving} maxLength={4096} placeholder="Leave blank to keep the saved key" required={configuration.configured && (destinationChanged || configuration.migration_required)} />
    </label>
    <p className="text-sm text-muted-foreground">Changing the host or public key requires a new secret key. Custom hosts must be registered by your deployment administrator.</p>
    {error && <p role="alert">{error}</p>}
    <div className="flex gap-2">
      <Button type="submit" disabled={disabled || saving}>{saving ? 'Saving…' : 'Save configuration'}</Button>
      <Button type="button" variant="secondary" disabled={disabled || saving || !configuration.configured} onClick={() => save(undefined, true)}>Remove saved key</Button>
      <Button type="button" variant="secondary" disabled={saving} onClick={() => onReload()}>Reload settings</Button>
    </div>
  </form>;
}

export function ExperimentalPostVerificationSettings ({ schema, hideTitle, disabled, onChanged }: { schema: AllSettings, hideTitle?: boolean, disabled?: boolean, onChanged?: () => void }) {
  return (
    <section className="space-y-6">
      {!hideTitle && <h2 className="text-lg font-medium">[Experimental] Post verifications</h2>}
      <SettingsField name="enable_post_verifications" item={schema.enable_post_verifications} onChanged={onChanged} disabled={disabled} />
      <SettingsField name="enable_post_verifications_for_widgets" item={schema.enable_post_verifications_for_widgets} onChanged={onChanged} disabled={disabled} />
    </section>
  );
}
