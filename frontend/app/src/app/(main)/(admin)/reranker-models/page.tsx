import { AdminPageHeading } from '@/components/admin-page-heading';
import { NextLink } from '@/components/nextjs/NextLink';
import RerankerModelsTable from '@/components/reranker/RerankerModelsTable';
import { PlusIcon } from 'lucide-react';

export default function Page () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'Reranker Models', docsUrl: '/docs/reranker-model' },
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
