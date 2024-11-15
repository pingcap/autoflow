import { cachedGetKnowledgeBaseById } from '@/app/(main)/(admin)/knowledge-bases/[id]/api';
import { KnowledgeBaseTabs } from '@/app/(main)/(admin)/knowledge-bases/[id]/tabs';
import { AdminPageHeading } from '@/components/admin-page-heading';
import type { ReactNode } from 'react';

export default async function KnowledgeBaseLayout ({ params, children }: { params: { id: string }, children: ReactNode }) {
  const id = parseInt(decodeURIComponent(params.id));
  const kb = await cachedGetKnowledgeBaseById(id);

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Knowledge Bases', url: '/knowledge-bases' },
          { title: kb.name },
        ]}
      />
      <KnowledgeBaseTabs id={id} />
      {children}
    </>
  );
}