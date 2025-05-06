'use client';

import { mutateEvaluationDataset, useEvaluationDataset } from '@/components/evaluations/hooks';
import { use, useTransition } from 'react';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { CreateEvaluationDatasetItemForm } from '@/components/evaluations/create-evaluation-dataset-item-form';
import { Loader2Icon } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function CreateEvaluationDatasetItemPage (props: { params: Promise<{ id: string }> }) {
  const params = use(props.params);
  const evaluationDatasetId = parseInt(decodeURIComponent(params.id));

  const { evaluationDataset } = useEvaluationDataset(evaluationDatasetId);

  const router = useRouter();
  const [transitioning, startTransition] = useTransition();

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: 'https://autoflow.tidb.ai/evaluation' },
          { title: 'Datasets', url: '/evaluation/datasets' },
          { title: evaluationDataset?.name ?? <Loader2Icon className="size-4 animate-spin repeat-infinite" />, url: `/evaluation/datasets/${evaluationDatasetId}` },
          { title: 'New Item' },
        ]}
      />
      <CreateEvaluationDatasetItemForm
        evaluationDatasetId={evaluationDatasetId}
        transitioning={transitioning}
        onCreated={() => {
          startTransition(() => {
            router.push(`/evaluation/datasets/${evaluationDatasetId}`);
            router.refresh();
            void mutateEvaluationDataset(evaluationDatasetId);
          });
        }}
      />
    </>
  );
}