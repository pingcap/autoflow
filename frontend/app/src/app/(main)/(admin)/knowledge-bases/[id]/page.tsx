import { DocumentsTable } from '@/components/documents/documents-table';

export default async function KnowledgeBasePage ({ params }: { params: { id: string } }) {
  const id = parseInt(decodeURIComponent(params.id));

  return (
    <>
      <DocumentsTable knowledgeBaseId={id} />
    </>
  );
}