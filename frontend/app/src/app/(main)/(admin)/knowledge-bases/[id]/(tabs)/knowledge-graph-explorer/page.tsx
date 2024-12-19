import { GraphEditor } from '@/components/graph/GraphEditor';

export default async function KnowledgeGraphExplorerPage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const id = parseInt(decodeURIComponent(params.id));

  return (
    <section className="space-y-2">
      <GraphEditor knowledgeBaseId={id} />
    </section>
  );
}