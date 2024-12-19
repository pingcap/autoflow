'use client';

import { type Document, listDocuments, type ListDocumentsTableFilters } from '@/api/documents';
import { deleteKnowledgeBaseDocument } from '@/api/knowledge-base';
import { actions } from '@/components/cells/actions';
import { datetime } from '@/components/cells/datetime';
import { link } from '@/components/cells/link';
import { mono } from '@/components/cells/mono';
import { DatasourceCell } from '@/components/cells/reference';
import { DataTableRemote } from '@/components/data-table-remote';
import { DocumentPreviewDialog } from '@/components/document-viewer';
import { DocumentsTableFilters } from '@/components/documents/documents-table-filters';
import { NextLink } from '@/components/nextjs/NextLink';
import { getErrorMessage } from '@/lib/errors';
import type { CellContext, ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { UploadIcon } from 'lucide-react';
import { useMemo, useState } from 'react';
import { toast } from 'sonner';

const helper = createColumnHelper<Document>();

const truncateUrl = (url: string, maxLength: number = 30): string => {
  if (!url || url.length <= maxLength) return url;
  const start = url.substring(0, maxLength / 2);
  const end = url.substring(url.length - maxLength / 2);
  return `${start}...${end}`;
};

const href = (cell: CellContext<any, string>) => <a className="underline" href={cell.getValue()} target="_blank">{truncateUrl(cell.getValue())}</a>;

const getColumns = (kbId: number) => [
  helper.accessor('id', { cell: mono }),
  helper.display({
    id: 'name', header: 'name', cell: ({ row }) =>
      <DocumentPreviewDialog
        title={row.original.source_uri}
        name={row.original.name}
        mime={row.original.mime_type}
        content={row.original.content}
      />,
  }),
  helper.display({
    id: 'chunks',
    header: 'Chunks',
    cell: link({
      url: row => `/knowledge-bases/${kbId}/documents/${row.id}/chunks`,
      text: () => 'View Chunks',
    }),
  }),
  helper.accessor('source_uri', { cell: href }),
  helper.accessor('mime_type', { cell: mono }),
  helper.accessor('data_source', { cell: ctx => <DatasourceCell {...ctx.getValue()} /> }),
  helper.accessor('created_at', { cell: datetime }),
  helper.accessor('updated_at', { cell: datetime }),
  helper.accessor('last_modified_at', { cell: datetime }),
  helper.display({
    id: 'op',
    header: 'Operations',
    cell: actions(row => [
      {
        key: 'delete-document',
        title: 'Delete',
        dangerous: {
          dialogTitle: `Continue to delete document "${row.name}"?`,
        },
        action: async (context) => {
          try {
            await deleteKnowledgeBaseDocument(kbId, row.id);
            context.table.reload?.();
            context.startTransition(() => {
              context.router.refresh();
            });
            context.setDropdownOpen(false);
            toast.success(`Successfully deleted document "${row.name}"`);
          } catch (e) {
            toast.error(`Failed to delete document "${row.name}"`, {
              description: getErrorMessage(e),
            });
            return Promise.reject(e);
          }
        },
      },
    ]),
  }),
] as ColumnDef<Document>[];

export function DocumentsTable ({ knowledgeBaseId }: { knowledgeBaseId: number }) {
  const [filters, setFilters] = useState<ListDocumentsTableFilters>({});

  const columns = useMemo(() => {
    return [...getColumns(knowledgeBaseId)];
  }, [knowledgeBaseId]);

  return (
    <DataTableRemote
      toolbar={((table) => (
        <div className="space-y-2">
          <NextLink
            href={`/knowledge-bases/${knowledgeBaseId}/data-sources/new?type=file`}
            variant="secondary"
          >
            <UploadIcon />
            Upload documents
          </NextLink>
          <DocumentsTableFilters table={table} onFilterChange={setFilters} />
        </div>
      ))}
      columns={columns}
      apiKey={knowledgeBaseId != null ? `api.datasource.${knowledgeBaseId}.documents` : 'api.documents.list'}
      api={(params) => listDocuments({ ...params, ...filters, knowledge_base_id: knowledgeBaseId })}
      apiDeps={[filters]}
      idColumn="id"
    />
  );
}

