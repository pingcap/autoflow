'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { EmbeddingModelsTable } from '@/components/embedding-models/EmbeddingModelsTable';
import { NextLink } from '@/components/nextjs/NextLink';
import { PlusIcon } from 'lucide-react';

export default function EmbeddingModelPage () {

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'Embedding Models', docsUrl: '/docs/embedding-model' },
        ]}
      />
      <NextLink href="/embedding-models/create">
        <PlusIcon className="size-4" />
        New Embedding Model
      </NextLink>
      <EmbeddingModelsTable />
    </>
  );
}
