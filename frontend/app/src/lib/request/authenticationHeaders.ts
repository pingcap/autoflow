export async function authenticationHeaders (): Promise<Record<string, string>> {
  return { 'X-CSRF-Guard': '1' };
}
