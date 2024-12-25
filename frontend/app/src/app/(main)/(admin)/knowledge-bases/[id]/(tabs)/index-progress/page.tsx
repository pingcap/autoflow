import { KnowledgeBaseIndexProgress } from '@/components/knowledge-base/knowledge-base-index';

export default async function KnowledgeBaseIndexProgressPage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const id = parseInt(decodeURIComponent(params.id));

  return (
    <section className="space-y-2">
      <h3 className="text-lg font-medium">Index Progress</h3>
      <KnowledgeBaseIndexProgress id={id} />
    </section>
  );
}