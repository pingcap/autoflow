import { DocumentsTable } from '@/components/documents/documents-table';

export default async function KnowledgeBasePage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const id = parseInt(decodeURIComponent(params.id));

  return (
    <>
      <DocumentsTable knowledgeBaseId={id} />
    </>
  );
}