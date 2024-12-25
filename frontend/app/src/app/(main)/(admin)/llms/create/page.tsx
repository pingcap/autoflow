'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { CreateLLMForm } from '@/components/llm/CreateLLMForm';
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
          { title: 'LLMs', url: '/llms', docsUrl: '/docs/llm' },
          { title: 'Create' },
        ]}
      />
      <CreateLLMForm
        transitioning={transitioning}
        onCreated={llm => {
          startTransition(() => {
            router.push(`/llms/${llm.id}`);
            router.refresh();
          });
        }}
      />
    </>
  );
}