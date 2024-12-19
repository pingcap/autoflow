import type { EvaluationTaskItem } from '@/api/evaluations';
import { AutoErrorMessagePopper } from '@/components/cells/error-message';
import { DocumentPreviewDialog } from '@/components/document-viewer';
import type { CellContext } from '@tanstack/react-table';
import { CircleCheckIcon, CircleDashedIcon, CircleXIcon, Loader2Icon } from 'lucide-react';
import { useMemo } from 'react';
import wcwidth from 'wcwidth';

// eslint-disable-next-line react/display-name
export const documentCell = (title: string, trimLength = 50, mime = 'text/markdown') => (context: CellContext<any, string | undefined | null>) => {
  const content = context.getValue();

  const splitIndex = useMemo(() => {
    if (!content) {
      return -1;
    }

    let n = 0;

    for (let i = 0; i < content.length; i++) {
      if (n < trimLength) {
        n += wcwidth(content[i]);
      } else {
        return i;
      }
    }

    return -1;
  }, [content, trimLength]);

  if (!content) {
    return '--';
  }

  if (splitIndex < 0) {
    return content;
  }

  return (
    <DocumentPreviewDialog
      title={title}
      name={content.slice(0, splitIndex) + '...'}
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
  return (
    <span className="inline-flex gap-1">
      {status === 'not_start' && <CircleDashedIcon className="text-muted-foreground flex-shrink-0 size-4" />}
      {status === 'cancel' && <CircleXIcon className="text-muted-foreground flex-shrink-0 size-4" />}
      {status === 'evaluating' && <Loader2Icon className="text-info flex-shrink-0 size-4 animate-spin repeat-infinite" />}
      {status === 'done' && <CircleCheckIcon className="text-success flex-shrink-0 size-4" />}
      {status === 'error' && <CircleXIcon className="text-destructive flex-shrink-0 size-4" />}
      <span className="text-accent-foreground">
        {status === 'not_start' ? 'Not started' : status === 'evaluating' ? 'Evaluating' : status === 'done' ? 'Done' : status === 'cancel' ? 'Cancelled' : 'Error:'}
      </span>
      {status === 'error' && <AutoErrorMessagePopper trimLength={28}>{error_msg}</AutoErrorMessagePopper>}
    </span>
  );
}
