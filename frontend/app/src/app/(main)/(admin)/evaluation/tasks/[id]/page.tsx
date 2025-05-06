'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { EvaluationTaskInfo } from '@/components/evaluations/evaluation-task-info';
import { EvaluationTaskItemsTable } from '@/components/evaluations/evaluation-task-items-table';
import { Loader2Icon } from 'lucide-react';
import { use } from 'react';
import { useEvaluationTask } from '@/components/evaluations/hooks';

export default function EvaluationTaskPage (props: { params: Promise<{ id: string }> }) {
  const params = use(props.params);
  const evaluationTaskId = parseInt(decodeURIComponent(params.id));

  const { evaluationTask } = useEvaluationTask(evaluationTaskId);

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: 'https://autoflow.tidb.ai/evaluation' },
          { title: 'Tasks', url: '/evaluation/tasks' },
          { title: evaluationTask?.name ?? <Loader2Icon className="size-4 animate-spin repeat-infinite" /> },
        ]}
      />
      <EvaluationTaskInfo evaluationTaskId={evaluationTaskId} />
      <EvaluationTaskItemsTable evaluationTaskId={evaluationTaskId} />
    </>
  );
}
