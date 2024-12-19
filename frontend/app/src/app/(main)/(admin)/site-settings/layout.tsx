'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { SecondaryNavigatorLayout, SecondaryNavigatorLink, SecondaryNavigatorList, SecondaryNavigatorMain } from '@/components/secondary-navigator-list';
import { type ReactNode } from 'react';

export default function SiteSettingsLayout ({ children }: { children: ReactNode }) {
  return (
    <div className="relative">
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Site Settings' },
        ]}
      />
      <SecondaryNavigatorLayout>
        <SecondaryNavigatorList>
          <SecondaryNavigatorLink pathname="/site-settings">
            Website
          </SecondaryNavigatorLink>
          <SecondaryNavigatorLink pathname="/site-settings/integrations">
            Integrations
          </SecondaryNavigatorLink>
          <SecondaryNavigatorLink pathname="/site-settings/custom_js">
            JS Widget
          </SecondaryNavigatorLink>
        </SecondaryNavigatorList>
        <SecondaryNavigatorMain className="px-2">
          {children}
        </SecondaryNavigatorMain>
      </SecondaryNavigatorLayout>
    </div>
  );
}
