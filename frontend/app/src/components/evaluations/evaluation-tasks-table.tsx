'use client';

import { type EvaluationTask, listEvaluationTasks } from '@/api/evaluations';
import { datetime } from '@/components/cells/datetime';
import { link } from '@/components/cells/link';
import { mono } from '@/components/cells/mono';
import { DataTableRemote } from '@/components/data-table-remote';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';

const helper = createColumnHelper<EvaluationTask>();

const columns = [
  helper.accessor('id', { header: 'ID', cell: mono }),
  helper.accessor('name', { header: 'Name', cell: link({ text: row => row.name, url: row => `/evaluation/tasks/${row.id}` }) }),
  helper.accessor('dataset_id', { header: 'Dataset', cell: link({ text: row => String(row.dataset_id), url: row => `/evaluation/datasets/${row.dataset_id}` }) }),
  helper.accessor('user_id', { header: 'User ID' }),
  helper.accessor('created_at', { header: 'Created At', cell: datetime }),
  helper.accessor('updated_at', { header: 'Updated At', cell: datetime }),
] as ColumnDef<EvaluationTask>[];

export function EvaluationTasksTable () {
  return (
    <DataTableRemote
      columns={columns}
      apiKey="api.evaluation.tasks.list"
      api={listEvaluationTasks}
      idColumn="id"
    />
  );
}
