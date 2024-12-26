import type { ChatMessageSource } from '@/api/chats';
import { useChatMessageField, useChatMessageStreamContainsState, useChatMessageStreamState } from '@/components/chat/chat-hooks';
import { ChatMessageController } from '@/components/chat/chat-message-controller';
import { AppChatStreamState } from '@/components/chat/chat-stream-state';
import { isNotFinished, parseSource } from '@/components/chat/utils';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';
import { LinkIcon, TextSearchIcon } from 'lucide-react';
import { useMemo } from 'react';

export function MessageContextSources ({ message }: { message: ChatMessageController | undefined }) {
  const sources = useChatMessageField(message, 'sources');
  const ongoing = useChatMessageStreamState(message);

  const shouldShow = useChatMessageStreamContainsState(message, AppChatStreamState.SEARCH_RELATED_DOCUMENTS);

  if (!shouldShow) {
    return null;
  }

  const uriSet = new Set<string>();
  const reducedContext = sources?.filter(source => {
    if (uriSet.has(source.source_uri)) {
      return false;
    }
    uriSet.add(source.source_uri);
    return true;
  });

  const animation = isNotFinished(ongoing);
  const hasSources = !!sources?.length;
  const empty = sources && sources.length === 0;

  return (
    <>
      <div className={cn('font-normal text-lg flex items-center gap-2 transition-opacity opacity-100', !hasSources && 'opacity-50')}>
        <TextSearchIcon size="1em" />
        Sources
      </div>
      {hasSources && <ScrollArea className="h-max w-full">
        <ul className="flex gap-2 py-4">
          {reducedContext?.map((source, index) => (
            <MessageContextSource key={source.source_uri} context={source} animation={animation} index={index} />
          ))}
        </ul>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>}
      {empty && ongoing?.state !== AppChatStreamState.SEARCH_RELATED_DOCUMENTS && <div className="text-muted-foreground">Empty</div>}
      {empty && ongoing?.state === AppChatStreamState.SEARCH_RELATED_DOCUMENTS && (
        <ul className="flex gap-2 py-4">
          <Skeleton className="rounded" style={{ width: 198, height: 52 }} />
          <Skeleton className="rounded" style={{ width: 198, height: 52 }} />
          <Skeleton className="rounded" style={{ width: 198, height: 52 }} />
        </ul>
      )}
    </>
  );
}

function MessageContextSource ({ index, animation, context }: { index: number, animation: boolean, context: ChatMessageSource }) {
  const source = useMemo(() => {
    return parseSource(context.source_uri);
  }, [context.source_uri]);

  return (
    <motion.li
      key={context.id}
      className="bg-card hover:bg-accent transition-colors w-[200px] overflow-hidden rounded-lg border text-xs"
      transition={{ delay: index * 0.1 }}
      initial={animation && { x: '-30%', opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
    >
      <a className="flex flex-col justify-between space-y-1 p-2 max-w-full h-full" href={context.source_uri} target="_blank">
        <div className="font-normal line-clamp-3 opacity-90">
          {context.name}
        </div>
        <div className="opacity-70 mt-auto mb-0">
          <LinkIcon size="1em" className="inline-flex mr-1" />
          {source}
        </div>
      </a>
    </motion.li>
  );
}

export function MessageContextSourceCard ({ title, href }: { title?: string, href?: string }) {
  const source = useMemo(() => {
    return parseSource(href);
  }, [href]);

  return (
    <a className="flex flex-col justify-between space-y-1 p-2 max-w-full h-full" href={href} target="_blank">
      <div className="font-normal line-clamp-3 opacity-90">
        {title}
      </div>
      <div className="opacity-70 mt-auto mb-0">
        <LinkIcon size="1em" className="inline-flex mr-1" />
        {source}
      </div>
    </a>
  );
}
