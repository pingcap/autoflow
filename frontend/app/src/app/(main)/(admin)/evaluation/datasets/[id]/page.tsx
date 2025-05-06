'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { EvaluationDatasetInfo } from '@/components/evaluations/evaluation-dataset-info';
import { EvaluationDatasetItemsTable } from '@/components/evaluations/evaluation-dataset-items-table';
import { Loader2Icon } from 'lucide-react';
import { NextLink } from '@/components/nextjs/NextLink';
import { Separator } from '@/components/ui/separator';
import { use } from 'react';
import { useEvaluationDataset } from '@/components/evaluations/hooks';

export default function EvaluationDatasetPage (props: { params: Promise<{ id: string }> }) {
  const params = use(props.params);
  const evaluationDatasetId = parseInt(decodeURIComponent(params.id));

  const { evaluationDataset, isLoading } = useEvaluationDataset(evaluationDatasetId);

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: 'https://autoflow.tidb.ai/evaluation' },
          { title: 'Datasets', url: '/evaluation/datasets' },
          { title: evaluationDataset?.name ?? <Loader2Icon className="size-4 animate-spin repeat-infinite" /> },
        ]}
      />
      <EvaluationDatasetInfo evaluationDatasetId={evaluationDatasetId} />
      <Separator className="space-y-6" />
      <NextLink href={`/evaluation/datasets/${evaluationDatasetId}/items/new`} disabled={isLoading}>
        New Item
      </NextLink>
      <EvaluationDatasetItemsTable evaluationDatasetId={evaluationDatasetId} />
    </>
  );
}
