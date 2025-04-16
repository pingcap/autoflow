'use client';

import { deleteEvaluationDatasetItem, type EvaluationDatasetItem, listEvaluationDatasetItems } from '@/api/evaluations';
import { actions } from '@/components/cells/actions';
import { datetime } from '@/components/cells/datetime';
import { link } from '@/components/cells/link';
import { metadataCell } from '@/components/cells/metadata';
import { DataTableRemote } from '@/components/data-table-remote';
import { documentCell, textChunksArrayCell } from '@/components/evaluations/cells';
import { type KeywordFilter, KeywordFilterToolbar } from '@/components/evaluations/keyword-filter-toolbar';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { useState } from 'react';

const helper = createColumnHelper<EvaluationDatasetItem>();

const columns = [
  helper.accessor('id', { header: 'ID', cell: link({ url: row => `/evaluation/datasets/${row.evaluation_dataset_id}/items/${row.id}` }) }),
  helper.accessor('query', { header: 'QUERY', cell: documentCell('Query') }),
  helper.accessor('reference', { header: 'REFERENCE', cell: documentCell('Reference') }),
  helper.accessor('retrieved_contexts', { header: 'RETRIEVED CONTEXTS', cell: textChunksArrayCell }),
  helper.accessor('extra', { header: 'EXTRA', cell: metadataCell }),
  helper.accessor('created_at', { header: 'CREATED AT', cell: datetime }),
  helper.accessor('updated_at', { header: 'UPDATED AT', cell: datetime }),
  helper.display({
    id: 'op',
    header: 'ACTIONS',
    cell: actions(row => ([
      {
        key: 'update',
        title: 'Update',
        action (context) {
          context.startTransition(() => {
            context.router.push(`/evaluation/datasets/${row.evaluation_dataset_id}/items/${row.id}`);
          });
        },
      },
      {
        key: 'delete',
        dangerous: {},
        title: 'Delete',
        async action (context) {
          await deleteEvaluationDatasetItem(row.evaluation_dataset_id, row.id);
          context.startTransition(() => {
            context.router.refresh();
          });
          context.setDropdownOpen(false);
          context.table.reload?.();
        },
      },
    ])),
  }),
] as ColumnDef<EvaluationDatasetItem>[];

export function EvaluationDatasetItemsTable ({ evaluationDatasetId }: { evaluationDatasetId: number }) {
  const [filter, setFilter] = useState<KeywordFilter>({ keyword: '' })
  return (
    <DataTableRemote
      columns={columns}
      toolbar={() => (
        <KeywordFilterToolbar onFilterChange={setFilter} />
      )}
      apiKey={`api.evaluation.datasets.${evaluationDatasetId}.all-items`}
      api={(page) => listEvaluationDatasetItems(evaluationDatasetId, { ...page, ...filter })}
      apiDeps={[filter.keyword]}
      idColumn="id"
    />
  );
}
