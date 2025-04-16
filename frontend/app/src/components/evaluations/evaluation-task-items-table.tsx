'use client';

import { type EvaluationTaskItem, listEvaluationTaskItems } from '@/api/evaluations';
import { datetime } from '@/components/cells/datetime';
import { metadataCell } from '@/components/cells/metadata';
import { mono } from '@/components/cells/mono';
import { percent } from '@/components/cells/percent';
import { DataTableRemote } from '@/components/data-table-remote';
import { documentCell, evaluationTaskStatusCell, textChunksArrayCell } from '@/components/evaluations/cells';
import { type KeywordFilter, KeywordFilterToolbar } from '@/components/evaluations/keyword-filter-toolbar';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { useState } from 'react';

const helper = createColumnHelper<EvaluationTaskItem>();

const columns = [
  helper.accessor('id', { header: 'ID', cell: mono }),
  helper.accessor('status', { header: 'STATUS', cell: evaluationTaskStatusCell, meta: { colSpan: context => context.row.original.status === 'error' ? 3 : 1 } }),
  helper.accessor('semantic_similarity', {
    header: 'SEMANTIC SIMILARITY',
    cell: context => percent(context, {
      colorStops: [
        { checkpoint: 0, color: 'hsl(var(--destructive))' },
        { checkpoint: 1 - 0.618, color: 'hsl(var(--destructive))' },
        { checkpoint: 0.5, color: 'hsl(var(--warning))' },
        { checkpoint: 0.618, color: 'hsl(var(--success))' },
        { checkpoint: 1, color: 'hsl(var(--success))' },
      ],
    }),
    meta: { colSpan: context => context.row.original.status === 'error' ? 0 : 1 }
  }),
  helper.accessor('factual_correctness', {
    header: 'FACTUAL CORRECTNESS',
    cell: context => percent(context, {
      colorStops: [
        { checkpoint: 0, color: 'hsl(var(--destructive))' },
        { checkpoint: 1 - 0.618, color: 'hsl(var(--destructive))' },
        { checkpoint: 0.5, color: 'hsl(var(--warning))' },
        { checkpoint: 0.618, color: 'hsl(var(--success))' },
        { checkpoint: 1, color: 'hsl(var(--success))' },
      ],
    }),
    meta: { colSpan: context => context.row.original.status === 'error' ? 0 : 1 }
  }),
  helper.accessor('query', { header: 'QUERY', cell: documentCell('Query') }),
  helper.accessor('chat_engine', { header: 'CHAT ENGINE' }),
  helper.accessor('reference', { header: 'REFERENCE', cell: documentCell('Reference') }),
  helper.accessor('response', { header: 'RESPONSE', cell: documentCell('Response') }),
  helper.accessor('retrieved_contexts', { header: 'RETRIEVED CONTEXTS', cell: textChunksArrayCell }),
  helper.accessor('extra', { header: 'EXTRA', cell: metadataCell }),
  helper.accessor('created_at', { header: 'CREATED AT', cell: datetime }),
  helper.accessor('updated_at', { header: 'UPDATED AT', cell: datetime }),
] as ColumnDef<EvaluationTaskItem>[];

export function EvaluationTaskItemsTable ({ evaluationTaskId }: { evaluationTaskId: number }) {
  const [filter, setFilter] = useState<KeywordFilter>({});
  return (
    <DataTableRemote
      columns={columns}
      toolbar={() => (
        <KeywordFilterToolbar onFilterChange={setFilter} />
      )}
      apiKey={`api.evaluation.tasks.${evaluationTaskId}.items.list`}
      api={(page) => listEvaluationTaskItems(evaluationTaskId, { ...page, ...filter })}
      apiDeps={[filter.keyword]}
      idColumn="id"
    />
  );
}
