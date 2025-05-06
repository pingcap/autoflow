'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { Loader2Icon } from 'lucide-react';
import { UpdateEvaluationDatasetItemForm } from '@/components/evaluations/update-evaluation-dataset-item-form';
import { use } from 'react';
import { useEvaluationDataset } from '@/components/evaluations/hooks';

export default function Page (props: { params: Promise<{ id: string, itemId: string }> }) {
  const params = use(props.params);
  const evaluationDatasetId = parseInt(decodeURIComponent(params.id));
  const evaluationDatasetItemId = parseInt(decodeURIComponent(params.itemId));

  const { evaluationDataset } = useEvaluationDataset(evaluationDatasetId);

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: 'https://autoflow.tidb.ai/evaluation' },
          { title: 'Datasets', url: '/evaluation/datasets' },
          { title: evaluationDataset?.name ?? <Loader2Icon className="size-4 animate-spin repeat-infinite" />, url: `/evaluation/datasets/${evaluationDatasetId}` },
          { title: `${evaluationDatasetItemId}` },
        ]}
      />
      <UpdateEvaluationDatasetItemForm
        evaluationDatasetId={evaluationDatasetId}
        evaluationDatasetItemId={evaluationDatasetItemId}
      />
    </>
  );
}
