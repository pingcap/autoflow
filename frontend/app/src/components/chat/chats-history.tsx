import { type Chat, deleteChat, listChats } from '@/api/chats';
import { useAuth } from '@/components/auth/AuthProvider';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { NextLink } from '@/components/nextjs/NextLink';
import { Button } from '@/components/ui/button';
import { SidebarMenuSkeleton } from '@/components/ui/sidebar';
import { cn } from '@/lib/utils';
import { TrashIcon } from 'lucide-react';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';
import useSWR from 'swr';

export function ChatsHistory () {
  const pathname = usePathname();
  const auth = useAuth();
  const user = auth.me;
  const { data: history, mutate, isLoading, isValidating } = useSWR('api.chats.list?size=8', () => listChats({ size: 8 }), {
    revalidateOnMount: false,
    keepPreviousData: true,
  });

  useEffect(() => {
    void mutate(() => undefined, { revalidate: true });
  }, [user?.id]);

  const isActive = (chat: Chat) => pathname === `/c/${chat.id}`;

  return (
    <div style={{ paddingLeft: 25 }}>
      {isLoading && (
        <>
          <SidebarMenuSkeleton />
          <SidebarMenuSkeleton />
          <SidebarMenuSkeleton />
        </>
      )}
      <ul className={cn('w-full overflow-hidden space-y-1 transition-opacity', isValidating && 'opacity-50')}>
        {history?.items.map(chat => (
          <li key={chat.id} className="flex gap-2 items-center">
            <NextLink href={`/c/${chat.id}`} data-active={isActive(chat) ? 'true' : undefined} variant={isActive(chat) ? 'secondary' : 'ghost'} className="flex-1 opacity-80 text-xs p-2 py-1.5 h-max font-light w-[86%] block whitespace-nowrap overflow-hidden overflow-ellipsis data-[active]:font-semibold transition-opacity text-left ellipsis">
              {chat.title}
            </NextLink>
            <DangerousActionButton
              asChild
              action={async () => {
                await deleteChat(chat.id).finally(() => mutate(history => history, { revalidate: true }));
              }}
              dialogTitle={`Are you sure to delete ${chat.title}?`}
              dialogDescription="This action cannot be undone."
            >
              <Button className="flex-shrink-0 w-max h-max rounded-full p-1 hover:bg-transparent" size="icon" variant="ghost">
                <TrashIcon className="w-3 h-3 opacity-20 hover:opacity-60" />
              </Button>
            </DangerousActionButton>
          </li>
        ))}
      </ul>
    </div>
  );
}