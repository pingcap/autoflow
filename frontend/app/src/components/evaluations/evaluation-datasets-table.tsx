'use client';

import { type EvaluationDataset, listEvaluationDatasets } from '@/api/evaluations';
import { datetime } from '@/components/cells/datetime';
import { link } from '@/components/cells/link';
import { mono } from '@/components/cells/mono';
import { DataTableRemote } from '@/components/data-table-remote';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';

const helper = createColumnHelper<EvaluationDataset>();

const columns = [
  helper.accessor('id', { header: 'ID', cell: mono }),
  helper.accessor('name', { header: 'Name', cell: link({ text: row => row.name, url: row => `/evaluation/datasets/${row.id}` }) }),
  helper.accessor('user_id', { header: 'User ID' }),
  helper.accessor('created_at', { header: 'Created At', cell: datetime }),
  helper.accessor('updated_at', { header: 'Updated At', cell: datetime }),
] as ColumnDef<EvaluationDataset>[];

export function EvaluationDatasetsTable () {
  return (
    <DataTableRemote
      columns={columns}
      apiKey="api.evaluation.datasets.list"
      api={listEvaluationDatasets}
      idColumn="id"
    />
  );
}
