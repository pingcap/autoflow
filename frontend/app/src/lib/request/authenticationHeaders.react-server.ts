import { cookies } from 'next/headers';

export async function authenticationHeaders (): Promise<Record<string, string>> {
  const k = await cookies();

  return { Cookie: k.toString() }; // Only work on rsc;
}
