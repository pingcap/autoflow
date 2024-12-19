import { headers } from 'next/headers';
import { SigninDialog } from './page.client';

export default async function Page () {
  const referer = (await headers()).get('Referer') ?? undefined;

  return (
    <SigninDialog callbackUrl={referer} />
  );
}
