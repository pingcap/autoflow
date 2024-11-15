'use client';

import { deleteDatasource, type DeprecatedDatasource, listDataSources } from '@/api/datasources';
import { actions } from '@/components/cells/actions';
import { link } from '@/components/cells/link';
import { DataTableRemote } from '@/components/data-table-remote';
import { LlmInfo } from '@/components/llm/LlmInfo';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { Trash2Icon } from 'lucide-react';

const helper = createColumnHelper<DeprecatedDatasource>();

const columns = [
  helper.accessor('name', { cell: link({ url: datasource => `/datasources/${datasource.id}` }) }),
  helper.accessor('data_source_type', {}),
  helper.accessor('llm_id', { cell: (ctx) => <LlmInfo className="justify-end" reverse id={ctx.getValue()} /> }),
  helper.accessor('build_kg_index', {}),
  helper.accessor('user_id', {}),
  helper.display({
    header: 'Actions',
    cell: actions(datasource => [
      {
        key: 'delete',
        title: 'Delete',
        icon: <Trash2Icon className="size-3" />,
        dangerous: {},
        action: async ({ table }) => {
          await deleteDatasource(datasource.id);
          table.reload?.();
        },
      },
    ]),
  }),
] as ColumnDef<DeprecatedDatasource>[];

export function DatasourceTable () {
  return (
    <DataTableRemote
      columns={columns}
      apiKey="api.datasources.list"
      api={listDataSources}
      idColumn="id"
    />
  );
}
