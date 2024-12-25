'use client';

import { type EvaluationDatasetItem, listEvaluationDatasetItems } from '@/api/evaluations';
import { datetime } from '@/components/cells/datetime';
import { metadataCell } from '@/components/cells/metadata';
import { mono } from '@/components/cells/mono';
import { DataTableRemote } from '@/components/data-table-remote';
import { documentCell, textChunksArrayCell } from '@/components/evaluations/cells';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';

const helper = createColumnHelper<EvaluationDatasetItem>();

const columns = [
  helper.accessor('id', { header: 'ID', cell: mono }),
  helper.accessor('query', { header: 'Query', cell: documentCell('Query', 25) }),
  helper.accessor('reference', { header: 'Reference', cell: documentCell('Reference', 25) }),
  helper.accessor('retrieved_contexts', { header: 'Retrieved Contexts', cell: textChunksArrayCell }),
  helper.accessor('extra', { header: 'extra', cell: metadataCell }),
  helper.accessor('created_at', { header: 'Created At', cell: datetime }),
  helper.accessor('updated_at', { header: 'Updated At', cell: datetime }),
] as ColumnDef<EvaluationDatasetItem>[];

export function EvaluationDatasetItemsTable ({ evaluationDatasetId }: { evaluationDatasetId: number }) {
  return (
    <DataTableRemote
      columns={columns}
      apiKey="api.evaluation.datasets.list"
      api={(page) => listEvaluationDatasetItems(evaluationDatasetId, page)}
      idColumn="id"
    />
  );
}
