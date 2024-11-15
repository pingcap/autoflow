'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { CreateEmbeddingModelForm } from '@/components/embedding-models/CreateEmbeddingModelForm';
import { useRouter } from 'next/navigation';
import { useTransition } from 'react';

export default function Page () {
  const router = useRouter();
  const [transitioning, startTransition] = useTransition();

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Embedding Models', url: '/embedding-models' },
          { title: 'Create' },
        ]}
      />
      <CreateEmbeddingModelForm
        transitioning={transitioning}
        onCreated={embeddingModel => {
          startTransition(() => {
            router.push(`/embedding-models/${embeddingModel.id}`);
          });
        }}
      />
    </>
  );
}