import { getLlm } from '@/api/llms';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { UpdateLlmForm } from '@/components/llm/UpdateLLMForm';

export default async function Page (props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const llm = await getLlm(parseInt(params.id));

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'LLMs', url: '/llms', docsUrl: 'https://autoflow.tidb.ai/llm' },
          { title: llm.name },
        ]}
      />
      <UpdateLlmForm llm={llm} />
    </>
  );
}
