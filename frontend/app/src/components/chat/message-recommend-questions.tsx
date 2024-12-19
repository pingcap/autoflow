import { getChatMessageRecommendedQuestions, reloadChatMessageRecommendedQuestions } from '@/api/chats';
import { useChatMessageField, useChatMessageStreamState, useCurrentChatController } from '@/components/chat/chat-hooks';
import type { ChatMessageController } from '@/components/chat/chat-message-controller';
import { isNotFinished } from '@/components/chat/utils';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { getErrorMessage } from '@/lib/errors';
import { cn } from '@/lib/utils';
import { PlusIcon, RefreshCwIcon } from 'lucide-react';
import { useState } from 'react';
import useSWR from 'swr';

export function MessageRecommendQuestions ({ assistant }: { assistant: ChatMessageController | undefined }) {
  const controller = useCurrentChatController();

  const state = useChatMessageStreamState(assistant);
  const finishedAt = useChatMessageField(assistant, 'finished_at');
  const shouldPerformRequest = !isNotFinished(state) && !!finishedAt;
  const { data, mutate, isLoading } = useSWR(assistant && shouldPerformRequest && `chat.messages.${assistant.id}.recommend-questions`, () => getChatMessageRecommendedQuestions(assistant!.id), { revalidateOnFocus: false });

  const [reloading, setReloading] = useState(false);
  const [reloadError, setReloadError] = useState<unknown>();

  const reload = () => {
    if (!assistant || !shouldPerformRequest) {
      return;
    }
    setReloading(true);
    reloadChatMessageRecommendedQuestions(assistant.id)
      .then(res => mutate(res, { revalidate: false }), error => setReloadError(error))
      .finally(() => {
        setReloading(false);
      });
  };

  if (!shouldPerformRequest) {
    return null;
  }

  return (
    <>
      <hr />
      <section className="space-y-2">
        <div className={cn('font-normal text-lg flex items-center gap-2 transition-opacity opacity-100')}>
          <div>
            Further questions
          </div>
          <button
            className={cn('ml-2 transition-colors text-primary/70 hover:text-primary', (isLoading || reloading) && 'text-muted-foreground')}
            disabled={isLoading || reloading}
            onClick={() => {
              reload();
            }}
          >
            <RefreshCwIcon className={cn('size-4 repeat-infinite', (isLoading || reloading) && 'animate-spin')} />
          </button>
        </div>
        {!!reloadError && (
          <Alert>
            <AlertTitle>Failed to reload recommended questions</AlertTitle>
            <AlertDescription>{getErrorMessage(reloadError)}</AlertDescription>
          </Alert>
        )}
        <ul className="">
          {isLoading && ['w-[70%]', 'w-[30%]', 'w-[50%]'].map(i => (
            <li key={i} className="last-of-type:border-b-0 border-b py-2 text-sm cursor-pointer transition-colors text-muted-foreground">
              <Skeleton className={cn('bg-muted h-3 py-0.5 w-60', i)} />
            </li>
          ))}
          {data?.map((q, i) => (
            <li key={i} className="text-sm last-of-type:border-b-0 border-b py-2">
              <button className="relative w-full text-left pr-8 cursor-pointer transition-colors text-muted-foreground hover:text-foreground" onClick={() => {
                void controller.post({
                  content: q,
                });
              }}>
                {q}
                <PlusIcon className="absolute stroke-2 size-4 right-2 top-1/2 -translate-y-1/2" />
              </button>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}