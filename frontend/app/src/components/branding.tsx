import type { PublicWebsiteSettings } from '@/api/site-settings';
import Link from 'next/link';

export function Branding ({ setting }: { setting: PublicWebsiteSettings }) {
  return (
    <Link className="flex items-center justify-start" href='/'>
      <img className="h-8 dark:hidden" src={setting.logo_in_light_mode} alt="logo" />
      <img className="h-8 hidden dark:block" src={setting.logo_in_dark_mode} alt="logo" />
    </Link>
  );
}
