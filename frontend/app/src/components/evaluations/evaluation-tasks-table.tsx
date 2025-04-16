'use client';

import { cancelEvaluationTask, type EvaluationTask, type EvaluationTaskWithSummary, listEvaluationTasks } from '@/api/evaluations';
import { actions } from '@/components/cells/actions';
import { datetime } from '@/components/cells/datetime';
import { link } from '@/components/cells/link';
import { mono } from '@/components/cells/mono';
import { DataTableRemote } from '@/components/data-table-remote';
import { mutateEvaluationTasks } from '@/components/evaluations/hooks';
import { type KeywordFilter, KeywordFilterToolbar } from '@/components/evaluations/keyword-filter-toolbar';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { useState } from 'react';

const helper = createColumnHelper<EvaluationTaskWithSummary>();

const columns = [
  helper.accessor('id', { header: 'ID', cell: mono }),
  helper.accessor('name', { header: 'NAME', cell: link({ text: row => row.name, url: row => `/evaluation/tasks/${row.id}` }) }),
  helper.accessor('dataset_id', { header: 'DATASET', cell: link({ text: row => String(row.dataset_id), url: row => `/evaluation/datasets/${row.dataset_id}` }) }),
  helper.accessor('user_id', { header: 'USER ID' }),
  helper.accessor('created_at', { header: 'CREATED AT', cell: datetime }),
  helper.accessor('updated_at', { header: 'UPDATED AT', cell: datetime }),
  helper.display({
    id: 'op',
    header: 'OPERATIONS',
    cell: actions(row => [
      {
        title: 'View',
        action: context => {
          context.startTransition(() => {
            context.router.push(`/evaluation/tasks/${row.id}`);
          });
        },
      },
      {
        title: 'Cancel',
        disabled: row.summary.not_start === 0,
        action: async (context) => {
          await cancelEvaluationTask(row.id);
          void mutateEvaluationTasks();
          context.setDropdownOpen(false);
        },
        dangerous: {},
      },
    ]),
  }),
] as ColumnDef<EvaluationTask>[];

export function EvaluationTasksTable () {
  const [filter, setFilter] = useState<KeywordFilter>({});
  return (
    <DataTableRemote
      columns={columns}
      toolbar={() => (
        <KeywordFilterToolbar onFilterChange={setFilter} />
      )}
      apiKey="api.evaluation.tasks.list"
      api={page => listEvaluationTasks({ ...page, ...filter })}
      apiDeps={[filter.keyword]}
      idColumn="id"
    />
  );
}
