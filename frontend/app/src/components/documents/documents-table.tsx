'use client';

import { link } from '@/components/cells/link';
import { type Document, listDocuments, type ListDocumentsTableFilters } from '@/api/documents';
import { deleteKnowledgeBaseDocument, rebuildKBDocumentIndex } from '@/api/knowledge-base';
import { actions } from '@/components/cells/actions';
import { datetime } from '@/components/cells/datetime';
import { mono } from '@/components/cells/mono';
import { DatasourceCell } from '@/components/cells/reference';
import { DataTableRemote } from '@/components/data-table-remote';
import { DocumentPreviewDialog } from '@/components/document-viewer';
import { DocumentsTableFilters } from '@/components/documents/documents-table-filters';
import { NextLink } from '@/components/nextjs/NextLink';
import { getErrorMessage } from '@/lib/errors';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { TrashIcon, UploadIcon, BlocksIcon, WrenchIcon, DownloadIcon } from 'lucide-react';
import { useMemo, useState } from 'react';
import { toast } from 'sonner';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

const helper = createColumnHelper<Document>();



const getColumns = (kbId: number) => [
  helper.accessor('id', { header: "ID", cell: mono }),
  helper.display({
    id: 'name', 
    header: 'NAME', 
    cell: ({ row }) =>
      <DocumentPreviewDialog
        title={row.original.source_uri}
        name={row.original.name}
        mime={row.original.mime_type}
        content={row.original.content}
      />,
  }),
  helper.accessor('source_uri', {
    header: "SOURCE URI",
    cell: link({
      icon: <DownloadIcon className="size-3" />,
      truncate: true,
    }),
  }),
  helper.accessor('data_source', { header: "DATA SOURCE", cell: ctx => <DatasourceCell {...ctx.getValue()} /> }),
  helper.accessor('updated_at', { header: "LAST UPDATED", cell: datetime }),
  helper.accessor('index_status', { header: "INDEX STATUS", cell: mono }),
  helper.display({
    id: 'op',
    header: 'ACTIONS',
    cell: actions(row => [
      {
        type: 'label',
        title: 'Actions',
      },
      {
        key: 'rebuild-index',
        title: 'Rebuild Index',
        icon: <WrenchIcon className="size-3" />,
        action: async (context) => {
          try {
            await rebuildKBDocumentIndex(kbId, row.id);
            context.table.reload?.();
            context.startTransition(() => {
              context.router.refresh();
            });
            context.setDropdownOpen(false);
            toast.success(`Successfully rebuild index for document "${row.name}"`);
          } catch (e) {
            toast.error(`Failed to rebuild index for document "${row.name}"`, {
              description: getErrorMessage(e),
            });
            return Promise.reject(e);
          }
        },
      },
      {
        key: 'view-chunks',
        title: 'View Chunks',
        icon: <BlocksIcon className="size-3" />,
        action: async (context) => {
          context.router.push(`/knowledge-bases/${kbId}/documents/${row.id}/chunks`);
        },
      },
      {
        type: 'separator',
      },
      {
        key: 'delete-document',
        title: 'Delete',
        icon: <TrashIcon className="size-3" />,
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
        <div className="space-y-1">
          {/* Top container - Search and Upload */}
          <div className="flex items-center justify-between py-1.5">
            <div></div>
            <NextLink
              href={`/knowledge-bases/${knowledgeBaseId}/data-sources/new?type=file`}
              variant="secondary"
              className="h-8 text-sm px-3"
            >
              <UploadIcon className="mr-2 size-3" />
              Upload
            </NextLink>
          </div>

          {/* Bottom container - Filters */}
          <div className="py-1.5 border-b">
            <DocumentsTableFilters table={table} onFilterChange={setFilters} />
          </div>
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

