'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { CreateRerankerForm } from '@/components/reranker/CreateRerankerForm';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';

export default function Page () {
  const router = useRouter();
  const [transitioning, startTransition] = useTransition();

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'Reranker Models', url: '/reranker-models', docsUrl: '/docs/reranker-model' },
          { title: 'Create' },
        ]}
      />
      <CreateRerankerForm
        transitioning={transitioning}
        onCreated={reranker => {
          startTransition(() => {
            router.push(`/reranker-models/${reranker.id}`);
            router.refresh();
          });

        }}
      />
    </>
  );
}