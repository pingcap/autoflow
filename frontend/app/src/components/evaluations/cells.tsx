import type { EvaluationTaskItem } from '@/api/evaluations';
import { DocumentPreviewDialog } from '@/components/document-viewer';
import type { CellContext } from '@tanstack/react-table';
import { CircleCheckIcon, CircleDashedIcon, CircleXIcon, Loader2Icon } from 'lucide-react';

// eslint-disable-next-line react/display-name
export const documentCell = (title: string, length = 25, mime = 'text/markdown') => (context: CellContext<any, string | undefined | null>) => {
  const content = context.getValue();
  if (!content) {
    return '--';
  }

  if (content.length < length) {
    return content;
  }

  return (
    <DocumentPreviewDialog
      title={title}
      name={content.slice(0, length) + '...'}
      mime={mime}
      content={content}
    />
  );
};

export const textChunksArrayCell = (context: CellContext<any, string[] | undefined | null>) => {
  return (context.getValue()?.length ?? '-') + ' Items';
};

export const evaluationTaskStatusCell = (context: CellContext<EvaluationTaskItem, EvaluationTaskItem['status']>) => {
  return <StatusCell row={context.row.original} />;
};

function StatusCell ({ row }: { row: EvaluationTaskItem }) {
  const { status, error_msg } = row;
  // TODO: popover error_msg
  return (
    <span className="inline-flex gap-1">
      {status === 'not_start' && <CircleDashedIcon className="text-muted-foreground flex-shrink-0 size-4" />}
      {status === 'evaluating' && <Loader2Icon className="text-info flex-shrink-0 size-4 animate-spin repeat-infinite" />}
      {status === 'done' && <CircleCheckIcon className="text-success flex-shrink-0 size-4" />}
      {status === 'error' && <CircleXIcon className="text-destructive flex-shrink-0 size-4" />}
      <span className="text-accent-foreground">
        {status === 'not_start' ? 'not start' : status === 'evaluating' ? 'evaluating' : status === 'done' ? 'done' : 'error'}
      </span>
    </span>
  );
}
