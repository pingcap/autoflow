import { AdminPageHeading } from '@/components/admin-page-heading';
import { ResourceNotFound } from '@/components/resource-not-found';

export default function NotFound () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: '/docs/evaluation' },
          { title: 'Datasets', url: '/evaluation/tasks' },
          { title: <span className="text-destructive">Not Found</span> },
        ]}
      />
      <ResourceNotFound resource="Evaluation Task" buttonHref="/evaluation/tasks" />
    </>
  );
}