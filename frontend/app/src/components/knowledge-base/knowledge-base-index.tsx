'use client';

import { type DatasourceKgIndexError, type DatasourceVectorIndexError } from '@/api/datasources';
import { listKnowledgeBaseKgIndexErrors, listKnowledgeBaseVectorIndexErrors, retryKnowledgeBaseAllFailedTasks } from '@/api/knowledge-base';
import { useKB } from '@/app/(main)/(admin)/knowledge-bases/[id]/context';
import { link } from '@/components/cells/link';
import { IndexProgressChart, IndexProgressChartPlaceholder } from '@/components/charts/IndexProgressChart';
import { TotalCard } from '@/components/charts/TotalCard';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { DataTableRemote } from '@/components/data-table-remote';
import { useKnowledgeBaseIndexProgress } from '@/components/knowledge-base/hooks';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { ArrowRightIcon, FileTextIcon, PuzzleIcon, RouteIcon } from 'lucide-react';
import Link from 'next/link';
import {useState} from "react";

export function KnowledgeBaseIndexProgress ({ id }: { id: number }) {
  const { index_methods } = useKB();
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
        {progress ? <IndexProgressChart title="Vector Index" data={progress.vector_index} /> : <IndexProgressChartPlaceholder title="Vector Index" />}
        {progress?.kg_index ? <IndexProgressChart title="Knowledge Graph Index" data={progress.kg_index} /> : <IndexProgressChartPlaceholder title="Knowledge Graph Index" />}
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
      <h3>Index Errors</h3>
      <Tabs defaultValue={showVectorIndexErrors ? 'vector-index-errors' : 'kg-index-errors'}>
        <div className="flex items-center">
          <TabsList>
            {showVectorIndexErrors && <TabsTrigger value="vector-index-errors">
              Vector Index
            </TabsTrigger>}
            {showKgIndexErrors && <TabsTrigger value="kg-index-errors">
              KnowledgeGraph Index
            </TabsTrigger>}
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
        {showVectorIndexErrors && <TabsContent value="vector-index-errors">
          <KBVectorIndexErrorsTable id={id} />
        </TabsContent>}
        {showKgIndexErrors && <TabsContent value="kg-index-errors">
          <KBKGIndexErrorsTable id={id} />
        </TabsContent>}
      </Tabs>
    </section>
  );
}

function KBVectorIndexErrorsTable ({ id }: { id: number }) {
  return (
    <DataTableRemote<DatasourceVectorIndexError, any>
      api={(params) => listKnowledgeBaseVectorIndexErrors(id, params)}
      apiKey={`datasources.${id}.vector-index-errors`}
      columns={vectorIndexErrorsColumns}
      idColumn="document_id"
    />
  );
}

function KBKGIndexErrorsTable ({ id }: { id: number }) {
  return (
    <DataTableRemote<DatasourceKgIndexError, any>
      api={(params) => listKnowledgeBaseKgIndexErrors(id, params)}
      apiKey={`datasources.${id}.kg-index-errors`}
      columns={kgIndexErrorsColumns}
      idColumn="chunk_id"
    />
  );
}

const vectorIndexErrorsHelper = createColumnHelper<DatasourceVectorIndexError>();
const vectorIndexErrorsColumns: ColumnDef<DatasourceVectorIndexError, any>[] = [
  vectorIndexErrorsHelper.display({
    header: 'Document', cell: ({ row }) => (
      <>
        {row.original.document_name}
        {' '}
        <span className="text-muted-foreground">#{row.original.document_id}</span>
      </>
    ),
  }),
  vectorIndexErrorsHelper.accessor('source_uri', { header: 'Source URI', cell: link({ url: row => row.source_uri, text: row => row.source_uri }) }),
  vectorIndexErrorsHelper.accessor('error', {
    cell: ({ getValue }) => <ErrorPopper>{getValue()}</ErrorPopper>,
  }),
];

const kgIndexErrorsHelper = createColumnHelper<DatasourceKgIndexError>();
const kgIndexErrorsColumns: ColumnDef<DatasourceKgIndexError, any>[] = [
  kgIndexErrorsHelper.accessor('source_uri', { header: 'Source URI', cell: link({ url: row => row.source_uri, text: row => row.source_uri }) }),
  kgIndexErrorsHelper.accessor('chunk_id', {}),
  kgIndexErrorsHelper.accessor('error', {
    cell: ({ getValue }) => <ErrorPopper>{getValue()}</ErrorPopper>,
  }),
];

function ErrorPopper ({ children }: { children: string | null }) {
  if (!children || children.length <= 25) {
    return children;
  }

  const shortcut = children.slice(0, 25);

  return (
    <HoverCard>
      <HoverCardTrigger>
        {shortcut}{'... '}
        <span className="text-muted-foreground">
          ({children.length + ' characters'})
        </span>
      </HoverCardTrigger>
      <HoverCardContent className="w-96 h-48">
        <div className="size-full overflow-scroll">
          <pre className="whitespace-pre">{children}</pre>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}