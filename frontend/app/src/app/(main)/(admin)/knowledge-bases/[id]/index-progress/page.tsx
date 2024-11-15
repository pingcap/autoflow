import { cachedGetKnowledgeBaseById } from '@/app/(main)/(admin)/knowledge-bases/[id]/api';
import { KBProvider } from '@/app/(main)/(admin)/knowledge-bases/[id]/context';
import { KnowledgeBaseIndexProgress } from '@/components/knowledge-base/knowledge-base-index';

export default async function KnowledgeBaseIndexProgressPage ({ params }: { params: { id: string } }) {
  const id = parseInt(decodeURIComponent(params.id));
  const kb = await cachedGetKnowledgeBaseById(id);

  return (
    <KBProvider value={kb}>
      <section className="space-y-2">
        <h3 className="text-lg font-medium">Index Progress</h3>
        <KnowledgeBaseIndexProgress id={kb.id} />
      </section>
    </KBProvider>
  );
}