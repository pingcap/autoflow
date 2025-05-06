'use client';

import { KnowledgeBaseCard, KnowledgeBaseCardPlaceholder } from '@/components/knowledge-base/knowledge-base-card';

import { AdminPageHeading } from '@/components/admin-page-heading';
import KnowledgeBaseEmptyState from '@/components/knowledge-base/empty-state';
import { NextLink } from '@/components/nextjs/NextLink';
import { useAllKnowledgeBases } from '@/components/knowledge-base/hooks';

export default function KnowledgeBasesPage () {
  const { data: knowledgeBases, isLoading } = useAllKnowledgeBases();

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Knowledge Bases', docsUrl: 'https://autoflow.tidb.ai/knowledge-base' },
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
