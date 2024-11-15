'use client';

import { type EmbeddingModel, listEmbeddingModels } from '@/api/embedding-models';
import { mono } from '@/components/cells/mono';
import { DataTableHeading } from '@/components/data-table-heading';
import { DataTableRemote } from '@/components/data-table-remote';
import { Badge } from '@/components/ui/badge';
import { buttonVariants } from '@/components/ui/button';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { PlusIcon } from 'lucide-react';
import Link from 'next/link';

export function EmbeddingModelsTable () {
  return (
    <DataTableRemote
      before={(
        <DataTableHeading>
          <span className="ml-auto" />
          <Link className={buttonVariants({ variant: 'default', className: 'gap-2' })} href="/embedding-models/create">
            <PlusIcon className="size-4" />
            New
          </Link>
        </DataTableHeading>
      )}
      columns={columns}
      apiKey="api.embedding-models.list"
      api={listEmbeddingModels}
      idColumn="id"
    />
  );
}

const helper = createColumnHelper<EmbeddingModel>();
const columns: ColumnDef<EmbeddingModel, any>[] = [
  helper.accessor('name', {
    header: 'Name',
    cell: ({ row }) => {
      const { id, name, is_default } = row.original;
      return (
        <Link className="flex gap-1 items-center underline" href={`/embedding-models/${id}`}>
          {is_default && <Badge>default</Badge>}
          {name}
        </Link>
      );
    },
  }),
  helper.display({
    header: 'Provider / Model',
    cell: ({ row }) => {
      const { model, provider } = row.original;
      return (
        <span className="flex gap-1 items-center">
          <Badge variant="secondary">{provider}</Badge>
          <Badge variant="outline">{model}</Badge>
        </span>
      );
    },
  }),
  helper.accessor('vector_dimension', { cell: mono }),
];
