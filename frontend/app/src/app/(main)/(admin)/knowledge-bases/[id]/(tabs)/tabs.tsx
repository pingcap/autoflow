'use client';

import { useKnowledgeBase } from '@/components/knowledge-base/hooks';
import { SecondaryNavigatorLink } from '@/components/secondary-navigator-list';

export function KnowledgeBaseTabs ({ knowledgeBaseId }: { knowledgeBaseId: number }) {
  const { knowledgeBase } = useKnowledgeBase(knowledgeBaseId);

  return (
    <>
      <SecondaryNavigatorLink pathname={`/knowledge-bases/${knowledgeBaseId}`}>
        Documents
        <span className="ml-auto text-xs font-normal text-muted-foreground">
          {knowledgeBase?.documents_total}
        </span>
      </SecondaryNavigatorLink>
      <SecondaryNavigatorLink pathname={`/knowledge-bases/${knowledgeBaseId}/data-sources`}>
        Data Sources
        <span className="ml-auto text-xs font-normal text-muted-foreground">
          {knowledgeBase?.data_sources_total}
        </span>
      </SecondaryNavigatorLink>
      <SecondaryNavigatorLink pathname={`/knowledge-bases/${knowledgeBaseId}/index-progress`}>
        Index Progress
      </SecondaryNavigatorLink>
      {/*<TabsTrigger*/}
      {/*  disabled={true}*/}
      {/*  value="retrieval-tester"*/}
      {/*  onClick={() => startTransition(() => {*/}
      {/*    router.push(`/knowledge-bases/${knowledgeBase.id}/retrieval-tester`);*/}
      {/*  })}*/}
      {/*>*/}
      {/*  Retrieval Tester*/}
      {/*</TabsTrigger>*/}
      <SecondaryNavigatorLink pathname={`/knowledge-bases/${knowledgeBaseId}/knowledge-graph-explorer`}>
        Graph Explorer
      </SecondaryNavigatorLink>
      <SecondaryNavigatorLink pathname={`/knowledge-bases/${knowledgeBaseId}/settings`}>
        Settings
      </SecondaryNavigatorLink>
    </>
  );
}