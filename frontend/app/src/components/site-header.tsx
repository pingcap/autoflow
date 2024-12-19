import type { PublicWebsiteSettings } from '@/api/site-settings';
import { Branding } from '@/components/branding';
import { SiteHeaderActions } from '@/components/site-header-actions';
import { SidebarTrigger } from '@/components/ui/sidebar';

export function SiteHeaderSmallScreen ({ setting }: { setting: PublicWebsiteSettings }) {
  return (
    <header className="md:hidden h-header px-2 sticky top-0 bg-background border-b z-10 flex gap-2 items-center">
      <Branding setting={setting} />
    </header>
  );
}

export function SiteHeaderLargeScreen ({ setting }: { setting: PublicWebsiteSettings }) {
  return (
    <div className="fixed top-0 right-2 md:top-4 md:right-4 z-10">
      <div className="flex gap-2 items-center">
        <SidebarTrigger className="md:hidden" />
        <SiteHeaderActions
          className="flex h-fit"
          social={{
            twitter: setting.social_twitter,
            github: setting.social_github,
            discord: setting.social_discord,
          }}
        />
      </div>
    </div>
  );
}