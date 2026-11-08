export function safeAnalyticsId (value: unknown): string {
  return typeof value === 'string' && /^(?:G-[A-Z0-9]{6,20}|UA-[0-9]{4,12}-[0-9]{1,4})$/.test(value) && !/\s/.test(value) ? value : '';
}
