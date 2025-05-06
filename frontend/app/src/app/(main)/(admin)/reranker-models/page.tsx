import { AdminPageHeading } from '@/components/admin-page-heading';
import { NextLink } from '@/components/nextjs/NextLink';
import { PlusIcon } from 'lucide-react';
import RerankerModelsTable from '@/components/reranker/RerankerModelsTable';

export default function Page () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'Reranker Models', docsUrl: 'https://autoflow.tidb.ai/reranker-model' },
        ]}
      />
      <NextLink href="/reranker-models/create">
        <PlusIcon className="size-4" />
        New Reranker Model
      </NextLink>
      <RerankerModelsTable />
    </>
  );
}
