import { getKnowledgeBaseDocumentChunks, type KnowledgeGraphDocumentChunk } from '@/api/knowledge-base';
import { DataTable } from '@/components/data-table';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import useSWR from 'swr';

export function DocumentChunksTable ({ knowledgeBaseId, documentId }: { knowledgeBaseId: number, documentId: number }) {
  const { data = [], isLoading } = useSWR(`api.knowledge-bases.${knowledgeBaseId}.document.${documentId}.chunks`, () => getKnowledgeBaseDocumentChunks(knowledgeBaseId, documentId), {
    revalidateOnFocus: false,
  });

  return (
    <DataTable
      columns={columns}
      data={data}
      loading={isLoading}
    />
  );
}

const columnsHelper = createColumnHelper<KnowledgeGraphDocumentChunk>();

const columns: ColumnDef<KnowledgeGraphDocumentChunk, any>[] = [
  columnsHelper.accessor('id', {}),
  columnsHelper.accessor('hash', {}),
  columnsHelper.accessor('text', {}),
];
