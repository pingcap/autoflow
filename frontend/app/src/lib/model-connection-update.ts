export interface ModelConnectionUpdate {
  config: Record<string, unknown>;
  credentials?: string | Record<string, unknown>;
}

function parseObject (text: string, label: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(`${label} must be a valid JSON object.`);
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return value as Record<string, unknown>;
}

export function modelConnectionUpdate (configText: string, credentialText: string, credentialType: string): ModelConnectionUpdate {
  const result: ModelConnectionUpdate = { config: parseObject(configText, 'Configuration') };
  if (credentialText.trim()) {
    result.credentials = credentialType === 'dict' ? parseObject(credentialText, 'Credentials') : credentialText;
  }
  return result;
}
