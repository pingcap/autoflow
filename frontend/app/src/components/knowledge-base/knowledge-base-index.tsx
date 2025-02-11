'use client';

import { actions } from '@/components/cells/actions';
import { type DatasourceKgIndexError, type DatasourceVectorIndexError } from '@/api/datasources';
import { listKnowledgeBaseKgIndexErrors, listKnowledgeBaseVectorIndexErrors, rebuildKBDocumentIndex, retryKnowledgeBaseAllFailedTasks } from '@/api/knowledge-base';
import { errorMessageCell } from '@/components/cells/error-message';
import { link } from '@/components/cells/link';
import { IndexProgressChart, IndexProgressChartPlaceholder } from '@/components/charts/IndexProgressChart';
import { TotalCard } from '@/components/charts/TotalCard';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { DataTableRemote } from '@/components/data-table-remote';
import { useKnowledgeBaseIndexProgress } from '@/components/knowledge-base/hooks';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { ArrowRightIcon, DownloadIcon, FileTextIcon, PuzzleIcon, RouteIcon, WrenchIcon } from 'lucide-react';
import Link from 'next/link';
import { toast } from 'sonner';
import { getErrorMessage } from '@/lib/errors';

export function KnowledgeBaseIndexProgress ({ id }: { id: number }) {
  const { progress, isLoading } = useKnowledgeBaseIndexProgress(id);

  return (
    <>
      <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
        <TotalCard
          title="Documents"
          icon={<FileTextIcon className="h-4 w-4 text-muted-foreground" />}
          total={progress?.documents.total}
          isLoading={isLoading}
        >
          <Link className="flex gap-2 items-center" href={`/knowledge-bases/${id}`}>All documents <ArrowRightIcon className="size-3" /></Link>
        </TotalCard>
        <TotalCard
          title="Chunks"
          icon={<PuzzleIcon className="h-4 w-4 text-muted-foreground" />}
          total={progress?.chunks.total}
          isLoading={isLoading}
        />
        <TotalCard
          title="Entities"
          icon={<RouteIcon className="h-4 w-4 text-muted-foreground" />}
          total={progress?.entities?.total || null}
          isLoading={isLoading}
        />
        <TotalCard
          title="Relationships"
          icon={<RouteIcon className="h-4 w-4 text-muted-foreground" />}
          total={progress?.relationships?.total || null}
          isLoading={isLoading}
        />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        {progress ? <IndexProgressChart title="Vector Index" data={progress.vector_index} label="Total Documents" /> : <IndexProgressChartPlaceholder title="Vector Index" label="Total Documents" />}
        {progress?.kg_index ? <IndexProgressChart title="Knowledge Graph Index" data={progress.kg_index} label="Total Chunks" /> : <IndexProgressChartPlaceholder title="Knowledge Graph Index" label="Total Chunks" />}
      </div>
      <KnowledgeBaseIndexErrors id={id} />
    </>
  );
}

export function KnowledgeBaseIndexErrors ({ id }: { id: number }) {
  const { progress, mutate } = useKnowledgeBaseIndexProgress(id);

  if (!progress) {
    return null;
  }
  const showVectorIndexErrors = !!progress.vector_index.failed;
  const showKgIndexErrors = !!progress.kg_index?.failed;

  if (!showVectorIndexErrors && !showKgIndexErrors) {
    return null;
  }

  return (
    <section className="space-y-4">
      <h3>Failed Tasks</h3>
      <Tabs defaultValue={showVectorIndexErrors ? 'vector-index-errors' : 'kg-index-errors'}>
        <div className="flex items-center">
          <TabsList>
            <TabsTrigger value="vector-index-errors">
              Vector Index
            </TabsTrigger>
            <TabsTrigger value="kg-index-errors">
              KnowledgeGraph Index
            </TabsTrigger>
          </TabsList>
          <DangerousActionButton
            className="ml-auto"
            action={async () => {
              await retryKnowledgeBaseAllFailedTasks(id);
              await mutate(undefined, { revalidate: true });
            }}
            dialogTitle="Retry failed tasks"
            dialogDescription="Are you sure to retry all failed tasks?"
          >
            Retry failed tasks
          </DangerousActionButton>

        </div>
        <TabsContent value="vector-index-errors">
          <KBVectorIndexErrorsTable kb_id={id} />
        </TabsContent>
        <TabsContent value="kg-index-errors">
          <KBKGIndexErrorsTable kb_id={id} />
        </TabsContent>
      </Tabs>
    </section>
  );
}

function KBVectorIndexErrorsTable ({ kb_id }: { kb_id: number }) {
  return (
    <DataTableRemote<DatasourceVectorIndexError, any>
      api={(params) => listKnowledgeBaseVectorIndexErrors(kb_id, params)}
      apiKey={`datasources.${kb_id}.vector-index-errors`}
      columns={getVectorIndexErrorsColumns(kb_id)}
      idColumn="document_id"
    />
  );
}

function KBKGIndexErrorsTable ({ kb_id }: { kb_id: number }) {
  return (
    <DataTableRemote<DatasourceKgIndexError, any>
      api={(params) => listKnowledgeBaseKgIndexErrors(kb_id, params)}
      apiKey={`datasources.${kb_id}.kg-index-errors`}
      columns={getKgIndexErrorsColumns(kb_id)}
      idColumn="chunk_id"
    />
  );
}

const vectorIndexErrorsHelper = createColumnHelper<DatasourceVectorIndexError>();
const getVectorIndexErrorsColumns = (kb_id: number): ColumnDef<DatasourceVectorIndexError, any>[] => {
  return [
    vectorIndexErrorsHelper.display({
      header: 'Document', cell: ({ row }) => (
        <>
          {row.original.document_name}
          {' '}
          <span className="text-muted-foreground">#{row.original.document_id}</span>
        </>
      ),
    }),
    vectorIndexErrorsHelper.accessor('source_uri', {
      header: 'Source URI',
      cell: link({ icon: <DownloadIcon className="size-3" />, truncate: true })
    }),
    vectorIndexErrorsHelper.accessor('error', {
      header: 'Error message',
      cell: errorMessageCell(),
    }),
    vectorIndexErrorsHelper.display({
      id: 'op',
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
              await rebuildKBDocumentIndex(kb_id, row.document_id);
              context.table.reload?.();
              context.startTransition(() => {
                context.router.refresh();
              });
              context.setDropdownOpen(false);
              toast.success(`Successfully rebuild index for document "${row.document_name}"`);
            } catch (e) {
              toast.error(`Failed to rebuild index for document "${row.document_name}"`, {
                description: getErrorMessage(e),
              });
              return Promise.reject(e);
            }
          },
        },
      ]),
    }),
  ]
};

const kgIndexErrorsHelper = createColumnHelper<DatasourceKgIndexError>();
const getKgIndexErrorsColumns = (kb_id: number): ColumnDef<DatasourceKgIndexError, any>[] => {
  return [
    kgIndexErrorsHelper.display({
      header: 'Document',
      cell: ({ row }) => (
      <>
        {row.original.document_name}
        {' '}
        <span className="text-muted-foreground">#{row.original.document_id}</span>
      </>
    ),
    }),
    kgIndexErrorsHelper.accessor('source_uri', {
      header: 'Source URI',
      cell: link({ icon: <DownloadIcon className="size-3" />, truncate: true })
    }),
    kgIndexErrorsHelper.accessor('chunk_id', { header: 'Chunk ID' }),
    kgIndexErrorsHelper.accessor('error', {
      header: 'Error message',
      cell: errorMessageCell(),
    }),
    kgIndexErrorsHelper.display({
      id: 'op',
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
              await rebuildKBDocumentIndex(kb_id, row.document_id);
              context.table.reload?.();
              context.startTransition(() => {
                context.router.refresh();
              });
              context.setDropdownOpen(false);
              toast.success(`Successfully rebuild knowledge graph index for document "${row.document_name}"`);
            } catch (e) {
              toast.error(`Failed to rebuild knowledge graph index for document "${row.document_name}"`, {
                description: getErrorMessage(e),
              });
              return Promise.reject(e);
            }
          },
        },
      ]),
    }),
  ]
};
