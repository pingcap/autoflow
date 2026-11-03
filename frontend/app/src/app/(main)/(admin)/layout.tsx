import { AdminPageLayout } from '@/components/admin-page-layout';
import { requireSuperuser } from '@/lib/auth';
import { type ReactNode } from 'react';

export default async function Layout ({ children }: { children: ReactNode }) {
  await requireSuperuser();
  return (
    <AdminPageLayout>
      {children}
    </AdminPageLayout>
  );
}
