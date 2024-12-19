'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import KnowledgeBaseEmptyState from '@/components/knowledge-base/empty-state';
import { useAllKnowledgeBases } from '@/components/knowledge-base/hooks';
import { KnowledgeBaseCard, KnowledgeBaseCardPlaceholder } from '@/components/knowledge-base/knowledge-base-card';
import { NextLink } from '@/components/nextjs/NextLink';

export default function KnowledgeBasesPage () {
  const { data: knowledgeBases, isLoading } = useAllKnowledgeBases();

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Knowledge Bases', docsUrl: '/docs/knowledge-base' },
        ]}
      />
      <NextLink href="/knowledge-bases/new">
        New Knowledge Base
      </NextLink>
      {
        isLoading
          ? <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4"><KnowledgeBaseCardPlaceholder /></div>
          : !!knowledgeBases?.length
            ? <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {knowledgeBases.map(kb => (
                <KnowledgeBaseCard key={kb.id} knowledgeBase={kb} />
              ))}
            </div>
            : <KnowledgeBaseEmptyState />
      }
    </>
  );
}

export const dynamic = 'force-dynamic';
