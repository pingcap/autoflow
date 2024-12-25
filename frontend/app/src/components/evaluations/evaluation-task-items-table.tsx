'use client';

import { type EvaluationTaskItem, listEvaluationTaskItems } from '@/api/evaluations';
import { datetime } from '@/components/cells/datetime';
import { metadataCell } from '@/components/cells/metadata';
import { mono } from '@/components/cells/mono';
import { percent } from '@/components/cells/percent';
import { DataTableRemote } from '@/components/data-table-remote';
import { documentCell, evaluationTaskStatusCell, textChunksArrayCell } from '@/components/evaluations/cells';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';
import { noPage } from '@/lib/zod';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';

const helper = createColumnHelper<EvaluationTaskItem>();

const columns = [
  helper.accessor('id', { header: 'ID', cell: mono }),
  helper.accessor('chat_engine', { header: 'ChatEngine' }),
  helper.accessor('status', { header: 'Status', cell: evaluationTaskStatusCell }),
  helper.accessor('query', { header: 'Query', cell: documentCell('Query', 25) }),
  helper.accessor('factual_correctness', { header: 'factual_correctness', cell: context => percent(context) }),
  helper.accessor('semantic_similarity', { header: 'semantic_similarity', cell: context => percent(context) }),
  helper.accessor('reference', { header: 'Reference', cell: documentCell('Reference', 25) }),
  helper.accessor('response', { header: 'Response', cell: documentCell('Response', 25) }),
  helper.accessor('retrieved_contexts', { header: 'Retrieved Contexts', cell: textChunksArrayCell }),
  helper.accessor('extra', { header: 'Extra', cell: metadataCell }),
  helper.accessor('error_msg', { header: 'Error Message', cell: ctx => <ErrorPopper>{ctx.getValue()}</ErrorPopper> }),
  helper.accessor('created_at', { header: 'Created At', cell: datetime }),
  helper.accessor('updated_at', { header: 'Updated At', cell: datetime }),
] as ColumnDef<EvaluationTaskItem>[];

export function EvaluationTaskItemsTable ({ evaluationTaskId }: { evaluationTaskId: number }) {
  return (
    <DataTableRemote
      columns={columns}
      apiKey={`api.evaluation.tasks.${evaluationTaskId}.items.list`}
      api={() => listEvaluationTaskItems(evaluationTaskId).then(res => noPage(res))}
      idColumn="id"
    />
  );
}

function ErrorPopper ({ children }: { children: string | null }) {
  if (!children || children.length <= 25) {
    return children;
  }

  const shortcut = children.slice(0, 25);

  return (
    <HoverCard>
      <HoverCardTrigger>
        {shortcut}{'... '}
        <span className="text-muted-foreground">
          ({children.length + ' characters'})
        </span>
      </HoverCardTrigger>
      <HoverCardContent className="w-96 h-48">
        <div className="size-full overflow-scroll">
          <pre className="whitespace-pre">{children}</pre>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}
