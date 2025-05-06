import { AdminPageHeading } from '@/components/admin-page-heading';
import { ResourceNotFound } from '@/components/resource-not-found';

export default function NotFound () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: 'https://autoflow.tidb.ai/evaluation' },
          { title: 'Datasets', url: '/evaluation/datasets' },
          { title: <span className="text-destructive">Not Found</span> },
        ]}
      />
      <ResourceNotFound resource="Evaluation Dataset" buttonHref="/evaluation/datasets" />
    </>
  );
}