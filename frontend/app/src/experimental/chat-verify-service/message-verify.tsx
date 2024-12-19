import { getVerify, isFinalVerifyState, isVisibleVerifyState, type MessageVerifyResponse, VerifyStatus } from '#experimental/chat-verify-service/api';
import { useChatMessageField, useCurrentChatController } from '@/components/chat/chat-hooks';
import type { ChatMessageController } from '@/components/chat/chat-message-controller';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Skeleton } from '@/components/ui/skeleton';
import { MessageVerifyResultMarkdown } from '@/experimental/chat-verify-service/message-verify-result-markdown';
import { getErrorMessage } from '@/lib/errors';
import { isServerError } from '@/lib/request';
import { cn } from '@/lib/utils';
import { AnimatePresence, motion } from 'framer-motion';
import Highlight from 'highlight.js/lib/core';
import sql from 'highlight.js/lib/languages/sql';
import { CheckCircle2Icon, ChevronDownIcon, CircleMinus, Loader2Icon, RefreshCwIcon, TriangleAlertIcon } from 'lucide-react';
import { type ReactElement, type ReactNode, useEffect, useState } from 'react';
import useSWR from 'swr';
import '@/components/code-theme.scss';

Highlight.registerLanguage('sql', sql);

export function MessageVerify ({ assistant }: { assistant: ChatMessageController | undefined }) {
  const [open, setOpen] = useState(false);
  const [notFound, setNotFound] = useState(false);

  const controller = useCurrentChatController();
  const messageVerifyUrl = useChatMessageField(assistant, 'post_verification_result_url');

  const { data: result, isLoading: isLoadingResult, error: pollError } = useSWR(
    messageVerifyUrl && `experimental.chat-message.post-verification.${messageVerifyUrl}`, () => getVerify(messageVerifyUrl!),
    {
      revalidateOnMount: true,
      revalidateOnFocus: false,
      errorRetryCount: 0,
      refreshInterval: data => {
        if (!data) {
          return 0;
        }
        return !isFinalVerifyState(data.status) ? 1000 : 0;
      },
      onError: err => {
        if (isServerError(err, 404)) {
          setNotFound(true);
        }

        console.log(err);
      },
    },
  );

  const canOpen = result ? isVisibleVerifyState(result.status) : false;
  const creating = (!result && isLoadingResult);
  const error: unknown = pollError;

  useEffect(() => {
    console.debug(`[message-verify]`, result);
  }, [result]);

  if (!messageVerifyUrl || notFound) { // Remove isSuperuser check
    return null;
  }

  return (
    <Collapsible
      open={canOpen ? open : false}
      onOpenChange={setOpen}
      className="p-2 border rounded-lg"
      disabled={!canOpen}
    >
      <CollapsibleTrigger asChild>
        <Button className="group gap-2 w-full break-words max-w-full text-wrap text-left h-max" variant="ghost">
          <MessageVerifyHeader result={result} creating={creating} error={error} />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent forceMount>
        <AnimatePresence>
          {open && result && <motion.div
            transition={{
              type: 'spring',
              duration: 0.25,
              bounce: false,
            }}
            initial={{ height: 0, opacity: 0, overflowY: 'hidden' }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0, overflowY: 'hidden' }}
            style={{ width: 'var(--radix-collapsible-content-width)' }}
            layout="size"
          >
            <MessageVerifyResultMarkdown content={result.runs_report ?? ''} />
          </motion.div>}
        </AnimatePresence>
      </CollapsibleContent>
      {!!error && <div className="px-4 text-destructive text-xs">{getErrorMessage(error) ?? defaultMessages.error}</div>}
      <div className="my-2 px-4 flex items-center flex-wrap justify-between">
        <div className="text-xs text-muted-foreground">
          Powered by <a className="underline font-bold" href="https://www.pingcap.com/tidb-cloud-serverless/?utm_source=tidb.ai&utm_medium=community" target="_blank">TiDB Serverless</a>
        </div>
        {result?.status === VerifyStatus.FAILED && controller.inputEnabled && (
          <Button
            size="sm"
            className="gap-1 text-xs px-2 py-1 h-max"
            variant="ghost"
            onClick={() => {
              controller.input = result.runs_report ?? '';
              controller.focusInput();
            }}
          >
            <RefreshCwIcon size="1em" />
            Regenerate with validation messages
          </Button>
        )}
      </div>
    </Collapsible>
  );
}

const defaultMessages = {
  'creating': 'Prepare to validate message...',
  'error': 'Unknown error',
  [VerifyStatus.CREATED]: 'Prepare to validate message...',
  [VerifyStatus.EXTRACTING]: 'Extracting SQL...',
  [VerifyStatus.VALIDATING]: 'Validation SQL...',
  [VerifyStatus.SUCCESS]: 'Message validation succeed.',
  [VerifyStatus.FAILED]: 'Message validation failed.',
  [VerifyStatus.SKIPPED]: 'Message validated skipped.',
};

const skippedIcon = <CircleMinus className="size-4" />;
const loadingIcon = <Loader2Icon className="size-4 animate-spin repeat-infinite" />;
const succeedIcon = <CheckCircle2Icon className="size-4 text-success" />;
const failedIcon = <TriangleAlertIcon className="size-4 text-warning" />;
const errorIcon = <TriangleAlertIcon className="size-4 text-destructive" />;

function MessageVerifyHeader ({ creating, error, result }: { creating?: boolean, error: unknown, result: MessageVerifyResponse | undefined }) {
  let icon: ReactElement<any> | undefined;
  let message: ReactNode | undefined;
  const indicatorVisible = result ? isVisibleVerifyState(result.status) : false;

  if (creating) {
    icon = <Skeleton className="block w-4 h-4 my-0.5 rounded-full bg-muted-foreground/30" />;
    message = <Skeleton className="inline-block w-48 h-4 my-0.5 rounded bg-muted-foreground/30" />;
  } else if (error) {
    icon = errorIcon;
    message = 'Failed to get post validation result.';
  } else {
    switch (result?.status) {
      case VerifyStatus.CREATED:
      case VerifyStatus.EXTRACTING:
      case VerifyStatus.VALIDATING:
        icon = loadingIcon;
        break;
      case VerifyStatus.SUCCESS:
        icon = succeedIcon;
        break;
      case VerifyStatus.FAILED:
        icon = failedIcon;
        break;
      case VerifyStatus.SKIPPED:
        icon = skippedIcon;
        break;
      default:
        icon = undefined;
        break;
    }
    message = result?.message ?? (result ? defaultMessages[result.status] : undefined) ?? 'Unknown validation state.';
  }

  return (
    <>
      {icon}
      {message}
      <ChevronDownIcon className={cn('size-4 ml-auto transition-transform group-data-[state=open]:rotate-180', indicatorVisible ? 'visible' : 'invisible')} />
    </>
  );
}
