import { type EvaluationDataset } from '@/api/evaluations';
import { DateFormat } from '@/components/date-format';
import { useEvaluationDataset } from '@/components/evaluations/hooks';
import { Skeleton } from '@/components/ui/skeleton';
import * as React from 'react';

export function EvaluationDatasetInfo ({ evaluationDatasetId }: { evaluationDatasetId: number }) {
  const { evaluationDataset } = useEvaluationDataset(evaluationDatasetId);

  if (evaluationDataset) {
    return <EvaluationDatasetInfoDisplay evaluationDataset={evaluationDataset} />;
  } else {
    return <EvaluationDatasetInfoSkeleton />;
  }
}

export function EvaluationDatasetInfoDisplay ({ evaluationDataset }: { evaluationDataset: EvaluationDataset }) {
  return (
    <div className="space-y-4">
      <div className="space-y-2 text-xs">
        <div>ID: {evaluationDataset.id}</div>
        <div>Created at: <DateFormat date={evaluationDataset.created_at} /></div>
        <div>Updated at: <DateFormat date={evaluationDataset.updated_at} /></div>
        <div>User ID: {evaluationDataset.user_id}</div>
      </div>
    </div>
  );
}

export function EvaluationDatasetInfoSkeleton ({}: {}) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[14em] h-[1em] rounded-sm" />
        </div>
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[14em] h-[1em] rounded-sm" />
        </div>
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[8em] h-[1em] rounded-sm" />
        </div>
      </div>
    </div>
  );
}

