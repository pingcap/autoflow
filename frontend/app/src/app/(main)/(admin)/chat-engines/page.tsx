import { AdminPageHeading } from '@/components/admin-page-heading';
import { ChatEnginesTable } from '@/components/chat-engine/chat-engines-table';
import { NextLink } from '@/components/nextjs/NextLink';

export default function ChatEnginesPage () {
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Chat Engines', docsUrl: '/docs/chat-engine' },
        ]}
      />
      <NextLink href="/chat-engines/new">New Chat Engine</NextLink>
      <ChatEnginesTable />
    </>
  );
}
