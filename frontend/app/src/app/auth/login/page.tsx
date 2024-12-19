import { Signin } from '@/components/signin';
import { headers } from 'next/headers';

export default async function Page() {
  const referer = (await headers()).get('Referer') ?? undefined;

  return (
    <div className="w-screen h-screen flex items-center justify-center bg-background">
      <div className="min-w-80 border rounded-lg p-4 bg-card space-y-4">
        <h1>
          Login
        </h1>
        <Signin callbackUrl={referer} />
      </div>
    </div>
  );
}
