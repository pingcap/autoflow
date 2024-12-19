import { useChatMessageStreamState } from '@/components/chat/chat-hooks';
import { ChatMessageController } from '@/components/chat/chat-message-controller';
import { MessageFeedback } from '@/components/chat/message-feedback';
import { useMessageFeedback } from '@/components/chat/use-message-feedback';
import { usePortalContainer } from '@/components/portal-provider';
import { Button } from '@/components/ui/button';
import { DialogTrigger } from '@/components/ui/dialog';
import { Tooltip, TooltipContent, TooltipPortal, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import copy from 'copy-to-clipboard';
import { CopyCheckIcon, CopyIcon, ThumbsDownIcon, ThumbsUpIcon } from 'lucide-react';
import { useState } from 'react';

export function MessageOperations ({ message }: { message: ChatMessageController }) {
  const streamState = useChatMessageStreamState(message);
  const { feedbackData, feedback: callFeedback, disabled } = useMessageFeedback(message.id);
  const [copied, setCopied] = useState(false);
  const [clicked, setClicked] = useState<'like' | 'dislike'>('like');
  const container = usePortalContainer();

  if (streamState) {
    return null;
  }
  return (
    <TooltipProvider>
      <div className="flex items-center gap-2">
        {/*<Button*/}
        {/*  size="sm"*/}
        {/*  className="gap-1 text-xs px-2 py-1 h-max"*/}
        {/*  variant="ghost"*/}
        {/*  onClick={() => controller.regenerate(message.id)}*/}
        {/*  disabled*/}
        {/*>*/}
        {/*  <RefreshCwIcon size="1em" />*/}
        {/*  Regenerate*/}
        {/*</Button>*/}

        <MessageFeedback
          initial={feedbackData}
          defaultAction={clicked}
          onFeedback={async (action, comment) => callFeedback(action, comment)}
        >
          {feedbackData
            ? (<DialogTrigger asChild>
              <Button size="icon" variant="ghost" className="rounded-full w-7 h-7" disabled={disabled}>
                {feedbackData.feedback_type === 'like' ? <ThumbsUpIcon className="w-4 h-4 text-success" /> : <ThumbsDownIcon className="w-4 h-4 text-destructive" />}
              </Button>
            </DialogTrigger>)
            : (<>
              <Tooltip>
                <DialogTrigger asChild>
                  <TooltipTrigger asChild>
                    <Button size="icon" variant="ghost" className="rounded-full w-7 h-7" disabled={disabled} onClick={() => { setClicked('like'); }} aria-label="Like This Answer">
                      <ThumbsUpIcon className="w-4 h-4" />
                    </Button>
                  </TooltipTrigger>
                </DialogTrigger>
                <TooltipPortal container={container}>
                  <TooltipContent>
                    I like this answer :)
                  </TooltipContent>
                </TooltipPortal>
              </Tooltip>
              <Tooltip>
                <DialogTrigger asChild>
                  <TooltipTrigger asChild>
                    <Button size="icon" variant="ghost" className="rounded-full w-7 h-7" disabled={disabled} onClick={() => { setClicked('dislike'); }} aria-label="Dislike This Answer">
                      <ThumbsDownIcon className="w-4 h-4" />
                    </Button>
                  </TooltipTrigger>
                </DialogTrigger>
                <TooltipPortal container={container}>
                  <TooltipContent>
                    I dislike this answer :(
                  </TooltipContent>
                </TooltipPortal>
              </Tooltip>
            </>)}
        </MessageFeedback>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="icon"
              variant="ghost"
              className={cn('rounded-full w-7 h-7 transition-colors', copied && 'text-success hover:text-success hover:bg-success/10')}
              onClick={() => {
                setCopied(copy(message.content));
              }}
            >
              {copied
                ? <CopyCheckIcon className="w-4 h-4" />
                : <CopyIcon className="w-4 h-4" />}
            </Button>
          </TooltipTrigger>
          <TooltipPortal container={container}>
            <TooltipContent>
              {copied ? 'Copied!' : 'Copy'}
            </TooltipContent>
          </TooltipPortal>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}