'use client';

import { SiteSidebar } from '@/app/(main)/nav';
import { SiteHeaderLargeScreen, SiteHeaderSmallScreen } from '@/components/site-header';
import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar';
import { useSettingContext } from '@/components/website-setting-provider';
import { cn } from '@/lib/utils';
import { ReactNode, useState } from 'react';

export default function Layout ({ children }: {
  children: ReactNode
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const setting = useSettingContext();

  return (
    <>
      <SiteHeaderSmallScreen setting={setting} />
      <SidebarProvider open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <div className={cn('hidden md:block absolute pl-2.5 top-2.5 md:top-5 md:pl-5 z-10 transition-all ease-linear', sidebarOpen ? 'left-[--sidebar-width]' : 'left-0')}>
          <SidebarTrigger />
        </div>
        <SiteHeaderLargeScreen setting={setting} />
        <SiteSidebar setting={setting} />
        <main className="flex-1 overflow-x-hidden">
          {children}
        </main>
      </SidebarProvider>
    </>
  );
}
