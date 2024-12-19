import { AdminPageHeading } from '@/components/admin-page-heading';
import { LLMsTable } from '@/components/llm/LLMsTable';
import { NextLink } from '@/components/nextjs/NextLink';
import { PlusIcon } from 'lucide-react';

export default function Page () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'LLMs', docsUrl: '/docs/llm' },
        ]}
      />
      <NextLink href="/llms/create">
        <PlusIcon className="size-4" />
        New LLM
      </NextLink>
      <LLMsTable />
    </>
  );
}
