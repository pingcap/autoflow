import { AdminPageHeading } from '@/components/admin-page-heading';
import { CreateChatEngineForm } from '@/components/chat-engine/create-chat-engine-form';
import { getDefaultChatEngineOptions } from '@/api/chat-engines';

export default async function NewChatEnginePage () {
  const defaultOptions = await getDefaultChatEngineOptions();
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Chat Engines', docsUrl: 'https://autoflow.tidb.ai/chat-engine', url: '/chat-engines' },
          { title: 'New' },
        ]}
      />
      <CreateChatEngineForm defaultChatEngineOptions={defaultOptions} />
    </>
  );
}
