import { AdminPageHeading } from '@/components/admin-page-heading';
import { EvaluationTasksTable } from '@/components/evaluations/evaluation-tasks-table';
import { NextLink } from '@/components/nextjs/NextLink';

export default function EvaluationTasksPage () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: '/docs/evaluation' },
          { title: 'Tasks' },
        ]}
      />
      <NextLink href="/evaluation/tasks/create">New Evaluation Task</NextLink>
      <EvaluationTasksTable />
    </>
  );
}
