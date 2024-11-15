'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { CreateKnowledgeBaseForm } from '@/components/knowledge-base/create-knowledge-base-form';

export default function NewKnowledgeBasePage () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Knowledge Bases', url: '/knowledge-bases' },
          { title: 'New' },
        ]}
      />
      <CreateKnowledgeBaseForm />
    </>
  );
}
