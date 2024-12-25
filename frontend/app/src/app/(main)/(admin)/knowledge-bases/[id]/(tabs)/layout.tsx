'use client';;
import { use } from "react";

import { KnowledgeBaseTabs } from '@/app/(main)/(admin)/knowledge-bases/[id]/(tabs)/tabs';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { ArrowRightIcon } from '@/components/icons';
import { useKnowledgeBase } from '@/components/knowledge-base/hooks';
import { SecondaryNavigatorLayout, SecondaryNavigatorList, SecondaryNavigatorMain } from '@/components/secondary-navigator-list';
import { Loader2Icon } from 'lucide-react';
import Link from 'next/link';
import type { ReactNode } from 'react';

export default function KnowledgeBaseLayout(props: { params: Promise<{ id: string }>, children: ReactNode }) {
  const params = use(props.params);

  const {
    children
  } = props;

  const id = parseInt(decodeURIComponent(params.id));
  const { knowledgeBase } = useKnowledgeBase(id);

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Knowledge Bases', url: '/knowledge-bases', docsUrl: '/docs/knowledge-base' },
          {
            alert: knowledgeBase?.data_sources_total === 0 ? {
              variant: 'warning',
              content: <>
                <p>This Knowledge Base has no datasource.</p>
                <Link className="underline flex gap-2 items-center" href={`/knowledge-bases/${id}/data-sources/new`}>
                  Create Data Source
                  <ArrowRightIcon className="size-4" />
                </Link>
              </>,
            } : undefined,
            title: knowledgeBase?.name ?? <Loader2Icon className="size-4 animate-spin repeat-infinite" />,
          },
        ]}
      />
      <SecondaryNavigatorLayout>
        <SecondaryNavigatorList>
          <KnowledgeBaseTabs knowledgeBaseId={id} />
        </SecondaryNavigatorList>
        <SecondaryNavigatorMain className="space-y-4 px-2">
          {children}
        </SecondaryNavigatorMain>
      </SecondaryNavigatorLayout>
    </>
  );
}