'use client';

import { setDefault } from '@/api/commons';
import { listRerankers, type Reranker } from '@/api/rerankers';
import { actions } from '@/components/cells/actions';
import { DataTableRemote } from '@/components/data-table-remote';
import { Badge } from '@/components/ui/badge';
import { getErrorMessage } from '@/lib/errors';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import Link from 'next/link';
import { toast } from 'sonner';

export default function RerankerModelsTable () {
  return (
    <DataTableRemote
      columns={columns}
      apiKey="api.rerankers.list"
      api={listRerankers}
      idColumn="id"
    />
  );
}
const helper = createColumnHelper<Reranker>();
const columns: ColumnDef<Reranker, any>[] = [
  helper.accessor('id', {
    header: 'ID',
    cell: ({ row }) => row.original.id
  }),
  helper.accessor('name', {
    header: 'NAME',
    cell: ({ row }) => {
      const { id, name, is_default } = row.original;
      return (
        <Link className="flex gap-1 items-center underline" href={`/reranker-models/${id}`}>
          {is_default && <Badge>default</Badge>}
          {name}
        </Link>
      );
    },
  }),
  helper.display({
    header: 'PROVIDER / MODEL',
    cell: ({ row }) => {
      const { model, provider } = row.original;
      return (
        <>
          <strong>{provider}</strong>:<span>{model}</span>
        </>
      );
    },
  }),
  helper.accessor('top_n', {
    header: 'TOP N',
  }),
  helper.display({
    id: 'Operations',
    header: 'ACTIONS',
    cell: actions(row => ([
      {
        key: 'set-default',
        title: 'Set Default',
        disabled: row.is_default,
        action: async (context) => {
          try {
            await setDefault('reranker-models', row.id);
            context.table.reload?.();
            context.startTransition(() => {
              context.router.refresh();
            });
            context.setDropdownOpen(false);
            toast.success(`Successfully set default Reranker Model to ${row.name}.`);
          } catch (e) {
            toast.error(`Failed to set default Reranker Model to ${row.name}.`, {
              description: getErrorMessage(e),
            });
            throw e;
          }
        },
      },
    ])),
  }),
];
