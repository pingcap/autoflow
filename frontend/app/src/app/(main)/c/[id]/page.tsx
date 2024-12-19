import { type Chat, type ChatMessage, getChat } from '@/api/chats';
import { AutoScroll, ManualScrollVoter } from '@/components/auto-scroll';
import { Conversation } from '@/components/chat/conversation';
import { ErrorCard } from '@/components/error-card';
import { Button } from '@/components/ui/button';
import { auth } from '@/lib/auth';
import { isServerError } from '@/lib/request';
import type { Metadata } from 'next';
import { cookies } from 'next/headers';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { cache } from 'react';

const cachedGetChat = cache((id: string) => getChat(id)
  .then(res => {
    return res;
  })
  .catch(error => {
    if (isServerError(error, [404, 422 /* handle not UUID */])) {
      notFound();
    } else {
      return Promise.reject(error);
    }
  }));

export default async function ChatDetailPage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const id = params.id;
  const me = await auth();
  const bid = (await cookies()).get('bid')?.value;

  let chat: Chat | undefined;
  let messages: ChatMessage[];

  try {
    const detail = await cachedGetChat(id);
    chat = detail.chat;
    messages = detail.messages;
  } catch (error) {
    if (isServerError(error, 403)) {
      return (
        <div className="h-screen flex items-center justify-center xl:pr-side bg-accent">
          <ErrorCard
            title="Access denied"
            message="This chat is private"
          >
            <div className="flex gap-2 items-center mt-8">
              {!me && (
                <Button asChild>
                  <Link href="/auth/login">
                    Login to continue
                  </Link>
                </Button>
              )}
              <Button variant="ghost" asChild>
                <Link href="/">
                  Back to homepage
                </Link>
              </Button>
            </div>
          </ErrorCard>
        </div>
      );
    }
    throw error;
  }

  const shouldOpen = me
    ? me.id === chat?.user_id
    : bid === chat?.browser_id;

  return (
    <div className="xl:pr-side">
      <AutoScroll edgePixels={10}>
        <ManualScrollVoter />
        <Conversation
          key={chat?.id}
          chatId={id}
          open={shouldOpen}
          chat={chat}
          history={messages}
        />
      </AutoScroll>
    </div>
  );
}

export async function generateMetadata(props: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const params = await props.params;
  try {
    const chat = await cachedGetChat(params.id);

    return {
      title: chat.chat.title,
    };
  } catch (error) {
    if (isServerError(error, 403)) {
      return {};
    } else {
      throw error;
    }
  }
}

export const dynamic = 'force-dynamic';
