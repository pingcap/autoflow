'use client';

import { setDefault } from '@/api/commons';
import { type EmbeddingModel, listEmbeddingModels } from '@/api/embedding-models';
import { actions } from '@/components/cells/actions';
import { mono } from '@/components/cells/mono';
import { DataTableRemote } from '@/components/data-table-remote';
import { Badge } from '@/components/ui/badge';
import { getErrorMessage } from '@/lib/errors';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import Link from 'next/link';
import { toast } from 'sonner';

export function EmbeddingModelsTable () {
  return (
    <DataTableRemote
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
        <>
          <strong>{provider}</strong>:<span>{model}</span>
        </>
      );
    },
  }),
  helper.accessor('vector_dimension', { cell: mono }),
  helper.display({
    id: 'Operations',
    header: 'Operations',
    cell: actions(row => ([
      {
        key: 'set-default',
        title: 'Set Default',
        disabled: row.is_default,
        action: async (context) => {
          try {
            await setDefault('embedding-models', row.id);
            context.table.reload?.();
            context.startTransition(() => {
              context.router.refresh();
            });
            context.setDropdownOpen(false);
            toast.success(`Successfully set default Embedding Model to ${row.name}.`);
          } catch (e) {
            toast.error(`Failed to set default Embedding Model to ${row.name}.`, {
              description: getErrorMessage(e),
            });
            throw e;
          }
        },
      },
    ])),
  }),
];
