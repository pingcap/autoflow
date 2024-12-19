import { AdminPageHeading } from '@/components/admin-page-heading';
import { EvaluationDatasetsTable } from '@/components/evaluations/evaluation-datasets-table';
import { NextLink } from '@/components/nextjs/NextLink';

export default function EvaluationDatasetsPage () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Evaluation', docsUrl: '/docs/evaluation' },
          { title: 'Datasets' },
        ]}
      />
      <NextLink href="/evaluation/datasets/create">New Evaluation Dataset</NextLink>
      <EvaluationDatasetsTable />
    </>
  );
}
