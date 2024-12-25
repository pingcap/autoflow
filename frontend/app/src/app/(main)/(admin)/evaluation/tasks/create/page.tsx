'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { CreateEvaluationTaskForm } from '@/components/evaluations/create-evaluation-task-form';
import { mutateEvaluationTasks } from '@/components/evaluations/hooks';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';

export default function EvaluationTaskPage () {
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: '/docs/evaluation' },
          { title: 'Tasks', url: '/evaluation/tasks' },
          { title: 'Create' },
        ]}
      />
      <CreateEvaluationTaskForm
        transitioning={transitioning}
        onCreated={evaluationTask => {
          void mutateEvaluationTasks();
          startTransition(() => {
            router.push(`/evaluation/tasks/${evaluationTask.id}`);
            router.refresh();
          });
        }}
      />
    </>
  );
}
