'use client';;
import { use } from "react";

import { getKnowledgeBaseDocument, getKnowledgeBaseDocumentChunks } from '@/api/knowledge-base';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { DateFormat } from '@/components/date-format';
import { CodeInput } from '@/components/form/widgets/CodeInput';
import { useKnowledgeBase } from '@/components/knowledge-base/hooks';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2Icon } from 'lucide-react';
import useSWR from 'swr';

export default function DocumentChunksPage(props: { params: Promise<{ id: string, documentId: string }> }) {
  const params = use(props.params);
  const kbId = parseInt(decodeURIComponent(params.id));
  const documentId = parseInt(decodeURIComponent(params.documentId));
  const { knowledgeBase } = useKnowledgeBase(kbId);

  const { data: document } = useSWR(`api.knowledge-bases.${kbId}.documents.${documentId}`, () => getKnowledgeBaseDocument(kbId, documentId));

  const { data = [], isLoading } = useSWR(`api.knowledge-bases.${kbId}.documents.${documentId}.chunks`, () => getKnowledgeBaseDocumentChunks(kbId, documentId), {
    revalidateOnFocus: false,
  });

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Knowledge Bases', url: '/knowledge-bases', docsUrl: '/docs/knowledge-base' },
          { title: knowledgeBase?.name ?? <Loader2Icon className="size-4 animate-spin repeat-infinite" />, url: `/knowledge-bases/${kbId}` },
          { title: document?.name ?? <Loader2Icon className="size-4 animate-spin repeat-infinite" /> },
          { title: 'Chunks' },
        ]}
      />
      <div className="space-y-6 max-w-screen-sm">
        {data.map(chunk => (
          <Card key={chunk.id}>
            <CardHeader>
              <CardTitle className="text-base">
                {chunk.id}
              </CardTitle>
              <CardDescription className="text-xs">
                Hash: <code>{chunk.hash}</code>
                <br />
                Source: {chunk.source_uri}
                <br />
                Updated At: <DateFormat date={chunk.updated_at} />
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <section className="space-y-2">
                <h4>Meta</h4>
                <CodeInput disabled language="json" value={JSON.stringify(chunk.meta, undefined, 2)} />
              </section>
              <section className="space-y-2">
                <h4>Content</h4>
                <CodeInput className="h-96" disabled language="markdown" value={chunk.text} />
              </section>
              <section className="space-y-2">
                <h4>Embedding</h4>
                <CodeInput className="h-96" disabled language="json" value={JSON.stringify(chunk.embedding, undefined, 2)} />
              </section>
            </CardContent>
          </Card>
        ))}
      </div>
    </>
  );
}