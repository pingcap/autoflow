'use client';

import { type Document, listDocuments, type ListDocumentsTableFilters } from '@/api/documents';
import { datetime } from '@/components/cells/datetime';
import { mono } from '@/components/cells/mono';
import { DatasourceCell, KnowledgeBaseCell } from '@/components/cells/reference';
import { DataTableRemote } from '@/components/data-table-remote';
import { DocumentPreviewDialog } from '@/components/document-viewer';
import { DocumentsTableFilters } from '@/components/documents/documents-table-filters';
import { DocumentChunksTable } from '@/components/knowledge-base/document-chunks-table';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import type { CellContext, ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { useMemo, useState } from 'react';

const helper = createColumnHelper<Document>();

const truncateUrl = (url: string, maxLength: number = 30): string => {
  if (!url || url.length <= maxLength) return url;
  const start = url.substring(0, maxLength / 2);
  const end = url.substring(url.length - maxLength / 2);
  return `${start}...${end}`;
};

const href = (cell: CellContext<any, string>) => <a className="underline" href={cell.getValue()} target="_blank">{truncateUrl(cell.getValue())}</a>;

const getColumns = (kbId?: number) => [
  helper.accessor('id', { cell: mono }),
  helper.accessor('knowledge_base', { cell: ctx => <KnowledgeBaseCell {...ctx.getValue()} /> }),
  helper.display({ id: 'name', header: 'name', cell: ({ row }) =>
    <DocumentPreviewDialog
      title={row.original.source_uri}
      name={row.original.name}
      mime={row.original.mime_type}
      content={row.original.content}
    />
  }),
  helper.accessor('source_uri', { cell: href }),
  helper.accessor('mime_type', { cell: mono }),
  helper.accessor('data_source', { cell: ctx => <DatasourceCell {...ctx.getValue()} /> }),
  helper.accessor('created_at', { cell: datetime }),
  helper.accessor('updated_at', { cell: datetime }),
  helper.accessor('last_modified_at', { cell: datetime }),
  helper.display({
    id: 'op',
    header: 'action',
    cell: ({ row }) => (kbId ?? row.original.knowledge_base?.id) != null && (
      <Dialog>
        <DialogTrigger>
          <Button className='text-xs p-2' variant="ghost" size="sm">
            Chunks
          </Button>
        </DialogTrigger>
        <DialogContent className="max-w-[720px] w-full">
          <DialogHeader>
            <DialogTitle>
              Document Chunks
            </DialogTitle>
            <DialogDescription>
            </DialogDescription>
          </DialogHeader>
          <div className="w-full overflow-x-hidden">
            <DocumentChunksTable knowledgeBaseId={(kbId ?? row.original.knowledge_base?.id)!} documentId={row.original.id} />
          </div>
        </DialogContent>
      </Dialog>

    ),
  }),
] as ColumnDef<Document>[];

export function DocumentsTable ({ knowledgeBaseId }: { knowledgeBaseId?: number }) {
  const [filters, setFilters] = useState<ListDocumentsTableFilters>({});

  const columns = useMemo(() => {
    if (knowledgeBaseId != null) {
      const columns = [...getColumns(knowledgeBaseId)];
      columns.splice(1, 1);
      return columns;
    } else {
      return getColumns(knowledgeBaseId);
    }
  }, [knowledgeBaseId]);

  return (
    <DataTableRemote
      toolbar={((table) => (
        <DocumentsTableFilters table={table} onFilterChange={setFilters} />
      ))}
      columns={columns}
      apiKey={knowledgeBaseId != null ? `api.datasource.${knowledgeBaseId}.documents` : 'api.documents.list'}
      api={(params) => listDocuments({ ...params, ...filters, knowledge_base_id: knowledgeBaseId })}
      apiDeps={[filters]}
      idColumn="id"
    />
  );
}

