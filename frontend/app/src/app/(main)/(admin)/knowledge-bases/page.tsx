'use client';

import { listKnowledgeBases } from '@/api/knowledge-base';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { KnowledgeBaseCard } from '@/components/knowledge-base/knowledge-base-card';
import { NextLink } from '@/components/nextjs/NextLink';
import type { PaginationState } from '@tanstack/table-core';
import { useState } from 'react';
import useSWR from 'swr';
import KnowledgeBaseEmptyState from "@/components/knowledge-base/empty-state";

export default function KnowledgeBasesPage () {
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 10 });

  const { data, mutate, isLoading, isValidating } = useSWR(`api.knowledge-bases.list?page=${pagination.pageIndex}&size=${pagination.pageSize}`, () => listKnowledgeBases({ page: pagination.pageIndex + 1, size: pagination.pageSize }), {
    revalidateOnReconnect: false,
    revalidateOnFocus: false,
    focusThrottleInterval: 1000,
    keepPreviousData: true,
    onError: console.error,
  });

  return (
    <>
      <AdminPageHeading title="Knowledge Bases" />
      <NextLink href="/knowledge-bases/new">
        New Knowledge Base
      </NextLink>
      {
        (Array.isArray(data?.items) && data?.items.length > 0) ?
        <div className="grid grid-cols-3 gap-4">
          {data?.items.map(kb => (
            <KnowledgeBaseCard key={kb.id} knowledgeBase={kb}/>
          ))}
        </div> :
        <KnowledgeBaseEmptyState/>
      }
    </>
  );
}

export const dynamic = 'force-dynamic';
