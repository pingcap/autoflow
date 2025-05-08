import { getReranker } from '@/api/rerankers';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { UpdateRerankerForm } from '@/components/reranker/UpdateRerankerForm';

export default async function Page (props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const reranker = await getReranker(parseInt(params.id));

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'Reranker Models', url: '/reranker-models', docsUrl: 'https://autoflow.tidb.ai/reranker-model' },
          { title: reranker.name },
        ]}
      />
      <UpdateRerankerForm reranker={reranker} />
    </>
  );
}
