import { AdminPageLayout } from '@/components/admin-page-layout';
import { requireAuth } from '@/lib/auth';
import { type ReactNode } from 'react';

export default async function Layout ({ children }: { children: ReactNode }) {
  await requireAuth();
  return (
    <AdminPageLayout>
      {children}
    </AdminPageLayout>
  );
}