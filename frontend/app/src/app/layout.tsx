import { getPublicSiteSettings } from '@/api/site-settings';
import { getBootstrapStatus } from '@/api/system';
import { RootProviders } from '@/app/RootProviders';
import { experimentalFeatures } from '@/experimental/experimental-features';
import { auth } from '@/lib/auth';
import { GoogleAnalytics } from '@next/third-parties/google';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Script from 'next/script';
import { cache, type ReactNode } from 'react';

import './globals.css';
import './chart-theme.css';

const inter = Inter({ subsets: ['latin'] });

const cachedGetSettings = cache(getPublicSiteSettings);

export async function generateMetadata (): Promise<Metadata> {
  const { title, description } = await cachedGetSettings();
  return {
    title,
    description,
    icons: '/favicon.svg',
  };
};

export default async function RootLayout ({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  const [
    me,
    settings,
    bootstrapStatus,
  ] = await Promise.all([
    auth(),
    cachedGetSettings(),
    getBootstrapStatus(),
  ]);

  const _experimentalFeatures = experimentalFeatures();

  if (!settings.enable_post_verifications) {
    _experimentalFeatures.enable_message_post_verification = false;
  }

  return (
    <html lang="en" suppressHydrationWarning>
    <body className={inter.className}>
    <RootProviders me={me} settings={settings} bootstrapStatus={bootstrapStatus} experimentalFeatures={_experimentalFeatures}>
      {children}
    </RootProviders>
    {settings.ga_id && <GoogleAnalytics gaId={settings.ga_id} />}
    <Script async src="/widget.js" data-is-main-site="true" />
    </body>
    </html>
  );
}

export const dynamic = 'force-dynamic';
